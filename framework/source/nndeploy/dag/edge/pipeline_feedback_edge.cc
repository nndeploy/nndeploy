#include "nndeploy/dag/edge/pipeline_feedback_edge.h"

#include "nndeploy/dag/edge/data_packet.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<PipelineFeedbackEdge>>
    g_pipeline_edge_register(base::kEdgeTypePipelineFeedback);

PipelineFeedbackEdge::PipelineFeedbackEdge(base::ParallelType pt)
    : AbstractEdge(pt) {}

static inline const void* key_from_node(const Node* n) {
  return static_cast<const void*>(n);
}

base::Status PipelineFeedbackEdge::setQueueMaxSize(int queue_max_size) {
  if (queue_max_size <= 0) queue_max_size = 1;
  queue_max_size_ = queue_max_size;
  ring_.set_capacity(static_cast<uint64_t>(queue_max_size));
  return base::kStatusCodeOk;
}

base::Status PipelineFeedbackEdge::construct() {
  auto producers = getProducers();
  if (producers.size() != 1) {
    NNDEPLOY_LOGE(
        "PipelineFeedbackEdge requires exactly one producer, got %zu.\n",
        producers.size());
    return base::kStatusCodeErrorInvalidValue;
  }
  if (queue_max_size_ <= 0) setQueueMaxSize(1);
  auto consumers = getConsumers();
  for (auto* c : consumers) {
    ring_.register_consumer(key_from_node(c), true);
  }
  return base::kStatusCodeOk;
}

}  // namespace dag
}  // namespace nndeploy