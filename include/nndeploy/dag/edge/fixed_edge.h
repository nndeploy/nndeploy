#ifndef _NNDEPLOY_DAG_EDGE_EDGE_FIXED_EDGE_H_
#define _NNDEPLOY_DAG_EDGE_EDGE_FIXED_EDGE_H_

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

class FixedEdge : public AbstractEdge {
 public:
  FixedEdge(ParallelType paralle_type, std::initializer_list<Node *> producers,
            std::initializer_list<Node *> consumers);
  virtual ~FixedEdge();

  virtual base::Status set(device::Buffer *buffer, int index, bool is_external);
  virtual base::Status set(device::Buffer &buffer, int index, bool is_external);
  virtual base::Status create(device::Device *device,
                              const device::BufferDesc &desc, int index);
  virtual device::Buffer *getBuffer(const Node *comsumer);

  virtual base::Status set(device::Mat *mat, int index, bool is_external);
  virtual base::Status set(device::Mat &mat, int index, bool is_external);
  virtual base::Status create(device::Device *device,
                              const device::MatDesc &desc, int index,
                              const std::string &name);
  virtual device::Mat *getMat(const Node *comsumer);

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index, bool is_external);
  virtual base::Status set(cv::Mat &cv_mat, int index, bool is_external);
  virtual cv::Mat *getCvMat(const Node *comsumer);
#endif

  virtual base::Status set(device::Tensor *tensor, int index, bool is_external);
  virtual base::Status set(device::Tensor &tensor, int index, bool is_external);
  virtual base::Status create(device::Device *device,
                              const device::TensorDesc &desc, int index,
                              const std::string &name);
  virtual device::Tensor *getTensor(const Node *comsumer);

  virtual base::Status set(base::Param *param, int index, bool is_external);
  virtual base::Status set(base::Param &param, int index, bool is_external);
  virtual base::Param *getParam(const Node *comsumer);

  virtual base::Status set(void *anything, int index, bool is_external);
  virtual void *getAnything(const Node *comsumer);

  virtual int getIndex(const Node *comsumer);

 private:
  DataPacket *data_packet_;
};

}  // namespace dag
}  // namespace nndeploy

#endif
