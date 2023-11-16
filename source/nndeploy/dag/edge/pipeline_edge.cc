#include "nndeploy/dag/edge/pipeline_edge.h"

namespace nndeploy {
namespace dag {

PipelineEdge::PipelineEdge(ParallelType paralle_type,
                           std::initializer_list<Node *> producers,
                           std::initializer_list<Node *> consumers)
    : AbstractEdge(paralle_type, producers, consumers) {
  consumers_count_ = consumers.size();
  for (auto iter : consumers) {
    consumed_.insert({iter, consumers_count});
  }
}

PipelineEdge::~PipelineEdge() {
  consumed_.clear();
  for (auto iter : data_packets_) {
    delete iter.first;
    iter.clear();
  }
}

base::Status PipelineEdge::set(device::Buffer *buffer, int index,
                               bool is_external) {
  DataPacket *dp = new DataPacket();
  dp->set(buffer, index, is_external);
  data_packets_.push_back({dp, consumers_count_});
}
base::Status PipelineEdge::set(device::Buffer &buffer, int index,
                               bool is_external);
base::Status PipelineEdge::create(device::Device *device,
                                  const device::BufferDesc &desc, int index);
virtual device::Buffer *getBuffer(const Node *comsumer) {
  if (comsumer == nullptr &&) {
    if (consumed_.empty()) {
    } else {
    }
  }

  /**
   * @brief
   * #. 检测node是否合理
   * ## 不合理，报错
   * ## 合理
   * ### 检测是否有数据包
   * #### 没有数据包，等待
   * #### 有数据包，拿到数据
   * # 清除数据包
   */
  // check
  auto iter = consumed_.find(comsumer);
  if (iter == consumed_.end()) {
    return nullptr;
  } else {
    if (iter->second == 0) {
      return nullptr;
    } else {
      iter->second--;
      return data_packets_[iter->second].first->getBuffer();
    }
  }
}

base::Status PipelineEdge::set(device::Mat *mat, int index, bool is_external);
base::Status PipelineEdge::set(device::Mat &mat, int index, bool is_external);
base::Status PipelineEdge::create(device::Device *device,
                                  const device::MatDesc &desc, int index);
virtual device::Mat *getMat(const Node *comsumer);

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status PipelineEdge::set(cv::Mat *cv_mat, int index, bool is_external);
base::Status PipelineEdge::set(cv::Mat &cv_mat, int index, bool is_external);
virtual cv::Mat *getCvMat(const Node *comsumer);
#endif

base::Status PipelineEdge::set(device::Tensor *tensor, int index,
                               bool is_external);
base::Status PipelineEdge::set(device::Tensor &tensor, int index,
                               bool is_external);
base::Status PipelineEdge::create(device::Device *device,
                                  const device::TensorDesc &desc, int index);
virtual device::Tensor *getTensor(const Node *comsumer);

base::Status PipelineEdge::set(base::Param *param, int index, bool is_external);
base::Status PipelineEdge::set(base::Param &param, int index, bool is_external);
virtual base::Param *getParam(const Node *comsumer);

base::Status PipelineEdge::set(void *anything, int index, bool is_external);
virtual void *getAnything(const Node *comsumer);

virtual int getIndex(const Node *comsumer);

}  // namespace dag
}  // namespace nndeploy