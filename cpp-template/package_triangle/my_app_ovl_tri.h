#ifndef MY_APP_H
#define MY_APP_H

#include <vector>
#include <unordered_set>
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

    messages.InitChannels(thread_num());
    ctx.Init(messages, ctx.group_count);
    ctx.stage = 0;

    ForEach(inner_vertices, [&messages, &frag, &ctx](int tid, vertex_t v) {
      ctx.global_degree[v] = frag.GetLocalOutDegree(v);
      messages.SendMsgThroughOEdges<fragment_t, int>(frag, v,
                                                     ctx.global_degree[v], tid);
    });

    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();

    if (ctx.stage == 0) {
      ctx.stage = 1;
      messages.ParallelProcess<fragment_t, int>(
          thread_num(), frag,
          [&ctx](int tid, vertex_t u, int msg) { ctx.global_degree[u] = msg; });

      ForEach(inner_vertices, [&frag, &ctx, &messages, this](int tid, vertex_t v) {
        vid_t u_gid, v_gid;
        auto& nbr_vec = ctx.complete_neighbor[v];
        int degree = ctx.global_degree[v];
        nbr_vec.reserve(degree);
        auto es = frag.GetOutgoingAdjList(v);
        std::vector<vid_t> msg_vec;
        msg_vec.reserve(degree);
        for (auto& e : es) {
          auto u = e.get_neighbor();
          if (ctx.global_degree[u] < ctx.global_degree[v]) {
            nbr_vec.push_back(u);
            msg_vec.push_back(frag.Vertex2Gid(u));
          } else if (ctx.global_degree[u] == ctx.global_degree[v]) {
            u_gid = frag.Vertex2Gid(u);
            v_gid = frag.GetInnerVertexGid(v);
            if (v_gid > u_gid) {
              nbr_vec.push_back(u);
              msg_vec.push_back(u_gid);
            }
          }
        }
        messages.SendMsgThroughOEdges<fragment_t, std::vector<vid_t>>(
            frag, v, msg_vec, tid);
      });
      messages.ForceContinue();
    } else if (ctx.stage == 1) {
      ctx.stage = 2;
      messages.ParallelProcess<fragment_t, std::vector<vid_t>>(
          thread_num(), frag,
          [&frag, &ctx](int tid, vertex_t u, const std::vector<vid_t>& msg) {
            auto& nbr_vec = ctx.complete_neighbor[u];
            for (auto gid : msg) {
              vertex_t v;
              if (frag.Gid2Vertex(gid, v)) {
                nbr_vec.push_back(v);
              }
            }
          });

      std::vector<grape::DenseVertexSet<typename FRAG_T::vertices_t>>
          vertexsets(thread_num());

      ForEach(
          inner_vertices,
          [&vertexsets, &frag](int tid) {
            auto& ns = vertexsets[tid];
            ns.Init(frag.Vertices());
          },
          [&vertexsets, &ctx, &frag, this](int tid, vertex_t v) {
            auto& v0_nbr_set = vertexsets[tid];
            auto& v0_nbr_vec = ctx.complete_neighbor[v];
            int v_gid = frag.Vertex2Gid(v);
            int v_group = GetGroupId(v_gid, ctx.group_count);

            for (auto u : v0_nbr_vec) {
              int u_gid = frag.Vertex2Gid(u);
              int u_group = GetGroupId(u_gid, ctx.group_count);
              if (u_group != v_group) {
                v0_nbr_set.Insert(u);
              }
            }
            for (auto u : v0_nbr_vec) {
              auto& v1_nbr_vec = ctx.complete_neighbor[u];
              int u_gid = frag.Vertex2Gid(u);
              int u_group = GetGroupId(u_gid, ctx.group_count);

              if (u_group != v_group) {
                for (auto w : v1_nbr_vec) {
                  int w_gid = frag.Vertex2Gid(w);
                  int w_group = GetGroupId(w_gid, ctx.group_count);
                  if (v0_nbr_set.Exist(w) && w_group != v_group && w_group != u_group) {
                    grape::atomic_add(ctx.tricnt[v], 1);
//                    grape::atomic_add(ctx.tricnt[u], 1);
//                    grape::atomic_add(ctx.tricnt[w], 1);
                  }
                }
              }
            }
            v0_nbr_set.Clear();
          },
          [](int tid) {});

      ForEach(frag.OuterVertices(), [&messages, &frag, &ctx](int tid, vertex_t v) {
        if (ctx.tricnt[v] != 0) {
          messages.SyncStateOnOuterVertex<fragment_t, int>(frag, v,
                                                           ctx.tricnt[v], tid);
        }
      });
      messages.ForceContinue();
    } else if (ctx.stage == 2) {
      ctx.stage = 3;
      messages.ParallelProcess<fragment_t, int>(
          thread_num(), frag, [&ctx](int tid, vertex_t u, int deg) {
            grape::atomic_add(ctx.tricnt[u], deg);
          });
    } else {
      messages.ParallelProcess<fragment_t, int>(
          thread_num(), frag, [](int tid, vertex_t u, int) {});
    }
  }
};

}  // namespace gs

#endif  // MY_APP_H
