#ifndef MY_APP_H
#define MY_APP_H

#include <vector>
#include <unordered_map>
#include "grape/grape.h"
#include "my_app_context.h"

namespace gs {

template <typename FRAG_T>
class MyApp : public grape::ParallelAppBase<FRAG_T, MyAppContext<FRAG_T>>,
              public grape::ParallelEngine {
 public:
  INSTALL_PARALLEL_WORKER(MyApp<FRAG_T>, MyAppContext<FRAG_T>, FRAG_T);
  using vertex_t = typename fragment_t::vertex_t;
  using vid_t = typename fragment_t::vid_t;

  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kOnlyOut;

  int GetGroupId(vid_t gid, int group_count) const {
    return gid % group_count;
  }

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    int total_nodes = frag.GetTotalVerticesNum();

    messages.InitChannels(thread_num());
    ctx.Init(messages, ctx.group_count);  // Ensure the context is properly initialized

    ForEach(inner_vertices, [&messages, &frag, &ctx](int tid, vertex_t v) {
      ctx.global_degree[v] = frag.GetLocalOutDegree(v);
      auto es = frag.GetOutgoingAdjList(v);
      std::vector<vertex_t> neighbors;
      for (auto& e : es) {
        neighbors.push_back(e.get_neighbor());
      }
      ctx.complete_neighbor[v] = neighbors;
      messages.SendMsgThroughOEdges<fragment_t, std::vector<vertex_t>>(frag, v, neighbors, tid);
    });

    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    if (ctx.stage == 0) {
      ctx.stage = 1;

      messages.ParallelProcess<fragment_t, std::vector<vertex_t>>(
          thread_num(), frag,
          [&ctx](int tid, vertex_t v, const std::vector<vertex_t>& received_neighbors) {
            auto& local_neighbors = ctx.complete_neighbor[v];
            local_neighbors.insert(local_neighbors.end(), received_neighbors.begin(), received_neighbors.end());
          });

      ForEach(frag.InnerVertices(), [&ctx, this, &frag](int tid, vertex_t v) {
        int group_v = GetGroupId(frag.Vertex2Gid(v), ctx.group_count);

        bool all_same_group = true;
        for (auto u : ctx.complete_neighbor[v]) {
          int group_u = GetGroupId(frag.Vertex2Gid(u), ctx.group_count);
          if (group_u != group_v) {
            all_same_group = false;
            break;
          }
        }

        if (all_same_group) {
          ctx.tricnt[v] = 1;
        } else {
          ctx.tricnt[v] = 0;
        }
      });
      messages.ForceContinue();
    }
  }
};

}  // namespace gs

#endif  // MY_APP_H
