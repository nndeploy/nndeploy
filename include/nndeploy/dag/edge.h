
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
 * @brief 需要保证Edge和Mat、Tensor名字一致, 还有一些检测的工作要做
 *
 */
class NNDEPLOY_CC_API Edge : public base::NonCopyable {
 public:
  Edge() : name_(""), abstact_edge_(nullptr) {}
  Edge(const std::string &name) : name_(name), abstact_edge_(nullptr) {}
  virtual ~Edge() { delete abstact_edge_; }

  std::string getName() { return name_; }

  base::Status construct(ParallelType paralle_type,
                         std::initializer_list<Node *> producers,
                         std::initializer_list<Node *> consumers);

  base::Status set(device::Buffer *buffer, int index_, bool is_external = true);
  base::Status set(device::Buffer &buffer, int index_, bool is_external = true);
  base::Status create(device::Device *device, const device::BufferDesc &desc,
                      int index_);
  device::Buffer *getBuffer(const Node *comsumer);

  base::Status set(device::Mat *mat, int index_, bool is_external = true);
  base::Status set(device::Mat &mat, int index_, bool is_external = true);
  base::Status create(device::Device *device, const device::MatDesc &desc,
                      int index_);
  device::Mat *getMat(const Node *comsumer);

#ifdef ENABLE_NNDEPLOY_OPENCV
  base::Status set(cv::Mat *cv_mat, int index_, bool is_external = true);
  base::Status set(cv::Mat &cv_mat, int index_, bool is_external = true);
  cv::Mat *getCvMat(const Node *comsumer);
#endif

  base::Status set(device::Tensor *tensor, int index_, bool is_external = true);
  base::Status set(device::Tensor &tensor, int index_, bool is_external = true);
  base::Status create(device::Device *device, const device::TensorDesc &desc,
                      int index_);
  device::Tensor *getTensor(const Node *comsumer);

  base::Status set(base::Param *param, int index_, bool is_external = true);
  base::Status set(base::Param &param, int index_, bool is_external = true);
  base::Param *getParam(const Node *comsumer);

  base::Status set(void *anything, int index_, bool is_external = true);
  void *getAnything(const Node *comsumer);

  int getIndex(const Node *comsumer);

 private:
  std::string name_;
  AbstractEdge *abstact_edge_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EDGE_V2_H_ */
