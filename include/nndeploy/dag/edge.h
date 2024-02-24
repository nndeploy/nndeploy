
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
#include "nndeploy/dag/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

/**
 * @brief The names of Edge, Mat, and Tensor need to be consistent.
 *
 */
class NNDEPLOY_CC_API Edge : public base::NonCopyable {
 public:
  Edge() : name_(""), abstact_edge_(nullptr) {}
  Edge(const std::string &name) : name_(name), abstact_edge_(nullptr) {}
  virtual ~Edge() { delete abstact_edge_; }

  std::string getName() { return name_; }

  base::Status construct(ParallelType paralle_type,
                         std::vector<Node *> &producers,
                         std::vector<Node *> &consumers);

  base::Status set(device::Buffer *buffer, int index, bool is_external = true);
  base::Status set(device::Buffer &buffer, int index);
  device::Buffer *create(device::Device *device, const device::BufferDesc &desc,
                         int index);
  bool notifyWritten(device::Buffer *buffer);
  device::Buffer *getBuffer(const Node *node);
  device::Buffer *getGraphOutputBuffer();

  base::Status set(device::Mat *mat, int index, bool is_external = true);
  base::Status set(device::Mat &mat, int index);
  device::Mat *create(device::Device *device, const device::MatDesc &desc,
                      int index);
  bool notifyWritten(device::Mat *mat);
  device::Mat *getMat(const Node *node);
  device::Mat *getGraphOutputMat();

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

  /**
   * @brief
   *
   * @param node
   * @return true
   * @return false
   * @note Serving the pipeline parallel mode, update the data in the pipeline.
   */
  bool updateData(const Node *node);

  ParallelType getParallelType();

  /**
   * @brief
   *
   * @return true
   * @return false
   * @note Serving the pipeline parallel mode, terminate pipeline.
   */
  bool requestTerminate();

 private:
  std::string name_;
  AbstractEdge *abstact_edge_ = nullptr;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EDGE_V2_H_ */
