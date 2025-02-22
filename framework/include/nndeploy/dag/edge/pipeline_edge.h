#ifndef _NNDEPLOY_DAG_EDGE_PIPELINE_EDGE_H_
#define _NNDEPLOY_DAG_EDGE_PIPELINE_EDGE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/edge/abstract_edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

/**
 * @brief
 * 1. 只能一个线程得到整个图的结果
 * @note
 * # 问题一：对于多输入的节点，会不会产生输入不匹配的情况呢？
 * ## 答：不会，因为节点内部会一直等待多输入的数据都到来，才会开始执行。
 * # 问题二：当处理完一批数据后，线程池中线程是不是要释放呢？
 * ## 答：线程池中的线程会处于wait状态，基本不消耗cpu资源
 */
class PipelineEdge : public AbstractEdge {
 public:
  PipelineEdge(base::ParallelType paralle_type);
  virtual ~PipelineEdge();

  virtual base::Status construct();

  virtual base::Status set(device::Buffer *buffer, int index, bool is_external);
  virtual base::Status set(device::Buffer &buffer, int index);
  virtual device::Buffer *create(device::Device *device,
                                 const device::BufferDesc &desc, int index);
  virtual bool notifyWritten(device::Buffer *buffer);
  virtual device::Buffer *getBuffer(const Node *node);
  virtual device::Buffer *getGraphOutputBuffer();

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index, bool is_external);
  virtual base::Status set(cv::Mat &cv_mat, int index);
  virtual cv::Mat *create(int rows, int cols, int type, const cv::Vec3b& value,
                           int index);
  virtual bool notifyWritten(cv::Mat *cv_mat);
  virtual cv::Mat *getCvMat(const Node *node);
  virtual cv::Mat *getGraphOutputCvMat();
#endif

  virtual base::Status set(device::Tensor *tensor, int index, bool is_external);
  virtual base::Status set(device::Tensor &tensor, int index);
  virtual device::Tensor *create(device::Device *device,
                                 const device::TensorDesc &desc, int index,
                                 const std::string &name);
  virtual bool notifyWritten(device::Tensor *tensor);
  virtual device::Tensor *getTensor(const Node *node);
  virtual device::Tensor *getGraphOutputTensor();

  virtual base::Status takeDataPacket(DataPacket *data_packet);
  virtual bool notifyAnyWritten(void *anything);
  virtual DataPacket *getDataPacket(const Node *node);
  virtual DataPacket *getGraphOutputDataPacket();

  virtual base::Status set(base::Param *param, int index, bool is_external);
  virtual base::Status set(base::Param &param, int index);
  virtual bool notifyWritten(base::Param *param);
  virtual base::Param *getParam(const Node *node);
  virtual base::Param *getGraphOutputParam();

  virtual int getIndex(const Node *node);
  virtual int getGraphOutputIndex();

  virtual int getPosition(const Node *node);
  virtual int getGraphOutputPosition();

  virtual base::EdgeUpdateFlag update(const Node *node);

  virtual bool requestTerminate();

 private:
  PipelineDataPacket *getPipelineDataPacket(const Node *node);

 private:
  std::once_flag once_;
  std::mutex mutex_;
  std::condition_variable cv_;
  // 有多少个消费者
  int consumers_size_;
  // 数据包
  std::list<PipelineDataPacket *> data_packets_;
  // 每个消费者 消费 的数据包最新索引  与下面当前数据包的关系为该索引为其+1
  std::map<Node *, int> to_consume_index_;
  // 每个消费者 消费 的当前数据包
  std::map<Node *, PipelineDataPacket *> consuming_dp_;
};

}  // namespace dag
}  // namespace nndeploy

#endif
