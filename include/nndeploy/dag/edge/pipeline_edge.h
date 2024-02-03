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
#include "nndeploy/dag/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

/**
 * @brief
 * 1. 只能一个线程得到整个图的结果
 * 2. 每条边的生产者只能是一个节点
 * @note
 * # 问题一：对于多输入的节点，会不会产生输入不匹配的情况呢？
 * ## 答：不会，因为节点内部会一直等待多输入的数据都到来，才会开始执行。
 * # 问题二：当处理完一批数据后，线程池中线程是不是要释放呢？
 * ## 答：未知，需要进一步学习线程池原理
 * # 问题三：某个先发出的notify，另外一个线程后开始等待，这种情况怎么办
 * ## 答：未知
 * # 问题四：有没有单纯等待某个条件达成的方法
 */
class PipelineEdge : public AbstractEdge {
 public:
  PipelineEdge(ParallelType paralle_type, std::vector<Node *> &producers,
               std::vector<Node *> &consumers);
  virtual ~PipelineEdge();

  virtual base::Status set(device::Buffer *buffer, int index, bool is_external);
  virtual base::Status set(device::Buffer &buffer, int index);
  virtual device::Buffer *create(device::Device *device,
                                 const device::BufferDesc &desc, int index);
  virtual bool notifyWritten(device::Buffer *buffer);
  virtual device::Buffer *getBuffer(const Node *node);

  virtual base::Status set(device::Mat *mat, int index, bool is_external);
  virtual base::Status set(device::Mat &mat, int index);
  virtual device::Mat *create(device::Device *device,
                              const device::MatDesc &desc, int index,
                              const std::string &name);
  virtual bool notifyWritten(device::Mat *mat);
  virtual device::Mat *getMat(const Node *node);

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index, bool is_external);
  virtual base::Status set(cv::Mat &cv_mat, int index);
  virtual cv::Mat *getCvMat(const Node *node);
#endif

  virtual base::Status set(device::Tensor *tensor, int index, bool is_external);
  virtual base::Status set(device::Tensor &tensor, int index);
  virtual device::Tensor *create(device::Device *device,
                                 const device::TensorDesc &desc, int index,
                                 const std::string &name);
  virtual bool notifyWritten(device::Tensor *tensor);
  virtual device::Tensor *getTensor(const Node *node);

  virtual base::Status set(base::Param *param, int index, bool is_external);
  virtual base::Status set(base::Param &param, int index);
  virtual base::Param *getParam(const Node *node);

  virtual base::Status set(void *anything, int index, bool is_external);
  virtual void *getAnything(const Node *node);

  virtual int getIndex(const Node *node);

 private:
  PipelineDataPacket *getDataPacket(const Node *node);

  /**
   * @brief Get the Graph Output Edge Data Packet object
   *
   * @param node
   * @return PipelineDataPacket*
   * @note 用于获取图的输出节点的数据包
   */
  PipelineDataPacket *getGraphOutputEdgeDataPacket(const Node *node);
  /**
   * @brief Get the Consumer Node Edge Data Packet object
   *
   * @param node
   * @return PipelineDataPacket*
   * @note 用于获取消费者节点的数据包
   */
  PipelineDataPacket *getConsumerNodeEdgeDataPacket(const Node *node);

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  // 有多少个消费者
  int consumers_size_ = 0;
  // 数据包
  std::list<PipelineDataPacket *> data_packets_;
  // 每个消费者 消费 的数据包最新索引
  std::map<Node *, int> to_consume_index_;
  // 每个消费者 消费 的挡墙数据包
  std::map<Node *, PipelineDataPacket *> consuming_dp_;
};

}  // namespace dag
}  // namespace nndeploy

#endif
