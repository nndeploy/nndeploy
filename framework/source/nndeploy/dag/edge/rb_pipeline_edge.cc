#include "nndeploy/dag/edge/rb_pipeline_edge.h"

#include "nndeploy/dag/edge/data_packet.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<RbPipelineEdge>> g_pipeline_edge_register(
    base::kEdgeTypePipeline);

RbPipelineEdge::RbPipelineEdge(base::ParallelType paralle_type)
    : AbstractEdge(paralle_type), rb_(queue_max_size_) {}

RbPipelineEdge::~RbPipelineEdge() {
  consumers_size_ = 0;
  consuming_dp_.clear();
  consumer_id_.clear();
}

base::Status RbPipelineEdge::construct() {
  consumers_size_ = static_cast<int>(consumers_.size());
  rb_.construct([this]() {
    return std::make_unique<PipelineDataPacket>(consumers_size_);
  });

  for (const auto &c : consumers_) {
    std::size_t cid = rb_.register_consumer();
    consumer_id_[c] = cid;
  }
  return base::kStatusCodeOk;
}

base::Status RbPipelineEdge::setQueueMaxSize(int q) {
  queue_max_size_ = q;
  return base::kStatusCodeOk;
}

/**
 * @breif buffer interface
 */
base::Status RbPipelineEdge::set(device::Buffer *buffer, bool is_external) {
  PipelineDataPacket *pkt = rb_.reserve();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(pkt, "PipelineDataPacket is null.\n");

  base::Status status = pkt->set(buffer, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set buffer failed.\n");

  return base::kStatusCodeOk;
};

device::Buffer *RbPipelineEdge::create(device::Device *device,
                                       const device::BufferDesc &desc) {
  PipelineDataPacket *pkt = rb_.reserve();
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(pkt, "PipelineDataPacket is null.\n");

  device::Buffer *ret_value = pkt->create(device, desc);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      ret_value, "PipelineDataPacket create buffer failed.\n");
  return ret_value;
};

bool RbPipelineEdge::notifyWritten(device::Buffer *buffer) {
  rb_.push();
  return true;
};

device::Buffer *RbPipelineEdge::getBuffer(const Node *node) {
  auto *dp = consuming_dp_[const_cast<Node *>(node)];
  return dp ? dp->getBuffer() : nullptr;
};

device::Buffer *RbPipelineEdge::getGraphOutputBuffer() {
  auto *pkt = getGraphOutputDataPacket();
  return pkt ? pkt->getBuffer() : nullptr;
};

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status RbPipelineEdge::set(cv::Mat *cv_mat, bool is_external) {
  PipelineDataPacket *pkt = rb_.reserve();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(pkt, "PipelineDataPacket is null.\n");

  base::Status status = pkt->set(cv_mat, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set cvmat failed.\n");
}

cv::Mat *RbPipelineEdge::create(int rows, int cols, int type,
                                const cv::Vec3b &value) {
  PipelineDataPacket *pkt = rb_.reserve();
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(pkt, "PipelineDataPacket is null.\n");

  cv::Mat *ret_value = dp->create(rows, cols, type, value);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      ret_value, "PipelineDataPacket create cvmat failed.\n");
  return ret_value;
}

bool RbPipelineEdg::notifyWritten(cv::Mat *cv_mat) {
  rb_.push();
  return true;
}

cv::Mat *RbPipelineEdg::getCvMat(const Node *node) {
  auto *dp = consuming_dp_[const_cast<Node *>(node)];
  return dp ? dp->getCvMat() : nullptr;
}

cv::Mat *RbPipelineEdg::getGraphOutputCvMat() {
  auto *pkt = getGraphOutputDataPacket();
  return pkt ? pkt->CvMat() : nullptr;
}
#endif

DataPacket *RbPipelineEdge::getGraphOutputDataPacket() {
  return getDataPacket(nullptr);
}

DataPacket *RbPipelineEdge::getDataPacket(const Node *node) {
  auto it = consumer_id_.find(node);
  if (it == consumer_id_.end()) return nullptr;

  std::size_t cid = it->second;
  auto *pkt = rb_.peek(cid);
  if (!pkt) return nullptr;

  consuming_dp_[node] = pkt;
  return pkt;
}

}  // namespace dag
}  // namespace nndeploy