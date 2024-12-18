#ifndef MY_APP_CONTEXT_H_
#define MY_APP_CONTEXT_H_

#include <vector>

#include "grape/grape.h"

namespace gs {
/**
 * @brief Context for wedges.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class MyAppContext : public grape::VertexDataContext<FRAG_T, int> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;

  explicit MyAppContext(const FRAG_T& fragment)
      : grape::VertexDataContext<FRAG_T, int>(fragment, true),
        tricnt(this->data()) {}

  void Init(grape::ParallelMessageManager& messages, int group_count_param) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();

    global_degree.Init(vertices);
    complete_neighbor.Init(vertices);
    tricnt.SetValue(0);
    group_count = group_count_param;
    stage = 0;
  }

  typename FRAG_T::template vertex_array_t<int> global_degree;
  typename FRAG_T::template vertex_array_t<std::vector<vertex_t>> complete_neighbor;
  typename FRAG_T::template vertex_array_t<int>& tricnt;

  int group_count;
  int stage;
};
}  // namespace gs

#endif  // MY_APP_CONTEXT_H_
