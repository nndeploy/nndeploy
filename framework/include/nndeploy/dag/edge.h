
#ifndef _NNDEPLOY_DAG_EDGE_H_
#define _NNDEPLOY_DAG_EDGE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/edge/abstract_edge.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

/**
 * @brief The names of Edge, Mat, and Tensor need to be consistent.
 *
 */
class NNDEPLOY_CC_API Edge : public base::NonCopyable {
 public:
  Edge();
  Edge(const std::string &name);
  virtual ~Edge();

  std::string getName();

  /**
   * @brief Set the Parallel Type object
   *
   * @param paralle_type
   * @return base::Status
   * @note 在construct之前，调用该函数，内部创建出具体的edge
   */
  base::Status setParallelType(const base::ParallelType &paralle_type);
  base::ParallelType getParallelType();

  base::Status construct();

  base::Status set(device::Buffer *buffer, int index, bool is_external = true);
  base::Status set(device::Buffer &buffer, int index);
  device::Buffer *create(device::Device *device, const device::BufferDesc &desc,
                         int index);
  bool notifyWritten(device::Buffer *buffer);
  device::Buffer *getBuffer(const Node *node);
  device::Buffer *getGraphOutputBuffer();

#ifdef ENABLE_NNDEPLOY_OPENCV
  base::Status set(cv::Mat *cv_mat, int index, bool is_external = true);
  base::Status set(cv::Mat &cv_mat, int index);
  cv::Mat *getCvMat(const Node *node);
  cv::Mat *getGraphOutputCvMat();
#endif

  base::Status set(device::Tensor *tensor, int index, bool is_external = true);
  base::Status set(device::Tensor &tensor, int index);
  device::Tensor *create(device::Device *device, const device::TensorDesc &desc,
                         int index);
  bool notifyWritten(device::Tensor *tensor);
  device::Tensor *getTensor(const Node *node);
  device::Tensor *getGraphOutputTensor();

  base::Status set(base::Param *param, int index, bool is_external = true);
  base::Status set(base::Param &param, int index);
  base::Param *getParam(const Node *node);
  base::Param *getGraphOutputParam();

  base::Status set(void *anything, int index, bool is_external = true);
  void *getAnything(const Node *node);
  void *getGraphOutputAnything();

  int getIndex(const Node *node);
  int getGraphOutputIndex();

  int getPosition(const Node *node);
  int getGraphOutputPosition();

  base::EdgeUpdateFlag update(const Node *node);

  /**
   * @brief
   *
   * @return true
   * @return false
   * @note must be called after the graph is initialized
   */
  bool markGraphOutput();

  base::Status increaseProducers(std::vector<Node *> &producers);
  base::Status increaseConsumers(std::vector<Node *> &consumers);

  bool requestTerminate();

 private:
  std::string name_;
  AbstractEdge *abstact_edge_ = nullptr;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EDGE_V2_H_ */
