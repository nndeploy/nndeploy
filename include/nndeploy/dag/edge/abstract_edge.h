#ifndef _NNDEPLOY_DAG_EDGE_EDGE_ABSTRACT_EDGE_H_
#define _NNDEPLOY_DAG_EDGE_EDGE_ABSTRACT_EDGE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

class AbstractEdge : public base::NonCopyable {
 public:
  AbstractEdge(ParallelType paralle_type,
               std::initializer_list<Node *> producers,
               std::initializer_list<Node *> consumers);
  virtual ~AbstractEdge();

  virtual base::Status set(device::Buffer *buffer, int index_ = -1,
                           bool is_external = true) = 0;
  virtual base::Status set(device::Buffer &buffer, int index_ = -1,
                           bool is_external = true) = 0;
  virtual base::Status create(device::Device *device,
                              const device::BufferDesc &desc,
                              int index_ = -1) = 0;
  virtual device::Buffer *getBuffer(const Node *comsumer) = 0;

  virtual base::Status set(device::Mat *mat, int index_ = -1,
                           bool is_external = true) = 0;
  virtual base::Status set(device::Mat &mat, int index_ = -1,
                           bool is_external = true) = 0;
  virtual base::Status create(device::Device *device,
                              const device::MatDesc &desc, int index_ = -1) = 0;
  virtual device::Mat *getMat(const Node *comsumer) = 0;

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index_ = -1,
                           bool is_external = true) = 0;
  virtual base::Status set(cv::Mat &cv_mat, int index_ = -1,
                           bool is_external = true) = 0;
  virtual cv::Mat *getCvMat(const Node *comsumer) = 0;
#endif

  virtual base::Status set(device::Tensor *tensor, int index_ = -1,
                           bool is_external = true) = 0;
  virtual base::Status set(device::Tensor &tensor, int index_ = -1,
                           bool is_external = true) = 0;
  virtual base::Status create(device::Device *device,
                              const device::TensorDesc &desc, int index_ = -1);
  virtual device::Tensor *getTensor(const Node *comsumer) = 0;

  virtual base::Status set(base::Param *param, bool is_external = true,
                           int pts = -1) = 0;
  virtual base::Status set(base::Param &param, bool is_external = true,
                           int pts = -1) = 0;
  virtual base::Param *getParam(const Node *comsumer) = 0;

  virtual base::Status set(void *anything, bool is_external = true,
                           int pts = -1) = 0;
  virtual void *getAnything(const Node *comsumer) = 0;

  virtual int getIndex(const Node *comsumer) = 0;

 private:
  ParallelType paralle_type_;
  std::vector<Node *> producers_;
  std::vector<Node *> consumers_;
};

AbstractEdge *createEdge(ParallelType paralle_type,
                         std::initializer_list<Node *> producers,
                         std::initializer_list<Node *> consumers);

}  // namespace dag
}  // namespace nndeploy

#endif /* E89A600B_4DA6_4278_B3D4_5AEBE24E8745 */
