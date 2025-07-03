#include "nndeploy/dag/edge/pipeline_edge.h"

#include "nndeploy/dag/edge/data_packet.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<PipelineEdge>> g_pipeline_edge_register(
    base::kEdgeTypePipeline);

PipelineEdge::PipelineEdge(base::ParallelType paralle_type)
    : AbstractEdge(paralle_type) {}

PipelineEdge::~PipelineEdge() {}

base::Status PipelineEdge::setQueueMaxSize(int max_size) {
  queue_max_size_ = max_size;
  return base::kStatusCodeOk;
}

base::Status PipelineEdge::construct() {
  consumers_size_ = static_cast<int>(consumers_.size());
  for (auto *n : consumers_) {
    consumer_id_[n] = rb_.register_consumer();
  }
  return base::kStatusCodeOk;
}

base::Status PipelineEdge::set(device::Buffer *buf, bool is_external) {
  auto *dp = new PipelineDataPacket(consumers_size_);
  this->increaseIndex();
  dp->setIndex(index_);
  base::Status st = dp->set(buf, is_external);
  if (st != base::kStatusCodeOk) {
    delete dp;
    return st;
  }
  PipelineDataPacket **slot;
  while ((slot = rb_.reserve()) == nullptr) {
    std::this_thread::yield();
  }
  *slot = dp;
  return base::kStatusCodeOk;
}

base::Status PipelineEdge::create()

}  // namespace dag
}  // namespace nndeploy