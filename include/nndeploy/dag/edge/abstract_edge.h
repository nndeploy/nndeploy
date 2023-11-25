#ifndef _NNDEPLOY_DAG_EDGE_ABSTRACT_EDGE_H_
#define _NNDEPLOY_DAG_EDGE_ABSTRACT_EDGE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/edge/data_packet.h"
#include "nndeploy/dag/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

class Node;

class AbstractEdge : public base::NonCopyable {
 public:
  AbstractEdge(ParallelType paralle_type,
               std::initializer_list<Node *> producers,
               std::initializer_list<Node *> consumers);
  virtual ~AbstractEdge();

  virtual base::Status set(device::Buffer *buffer, int index,
                           bool is_external) = 0;
  virtual base::Status set(device::Buffer &buffer, int index,
                           bool is_external) = 0;
  virtual base::Status create(device::Device *device,
                              const device::BufferDesc &desc, int index) = 0;
  virtual device::Buffer *getBuffer(const Node *comsumer) = 0;

  virtual base::Status set(device::Mat *mat, int index, bool is_external) = 0;
  virtual base::Status set(device::Mat &mat, int index, bool is_external) = 0;
  virtual base::Status create(device::Device *device,
                              const device::MatDesc &desc, int index,
                              const std::string &name) = 0;
  virtual device::Mat *getMat(const Node *comsumer) = 0;

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index, bool is_external) = 0;
  virtual base::Status set(cv::Mat &cv_mat, int index, bool is_external) = 0;
  virtual cv::Mat *getCvMat(const Node *comsumer) = 0;
#endif

  virtual base::Status set(device::Tensor *tensor, int index,
                           bool is_external) = 0;
  virtual base::Status set(device::Tensor &tensor, int index,
                           bool is_external) = 0;
  virtual base::Status create(device::Device *device,
                              const device::TensorDesc &desc, int index,
                              const std::string &name) = 0;
  virtual device::Tensor *getTensor(const Node *comsumer) = 0;

  virtual base::Status set(base::Param *param, int index, bool is_external) = 0;
  virtual base::Status set(base::Param &param, int index, bool is_external) = 0;
  virtual base::Param *getParam(const Node *comsumer) = 0;

  virtual base::Status set(void *anything, int index, bool is_external) = 0;
  virtual void *getAnything(const Node *comsumer) = 0;

  virtual int getIndex(const Node *comsumer) = 0;
  ParallelType getParallelType() { return paralle_type_; }

 protected:
  ParallelType paralle_type_;
  std::vector<Node *> producers_;
  std::vector<Node *> consumers_;
};

class EdgeCreator {
 public:
  virtual ~EdgeCreator(){};
  virtual AbstractEdge *createEdge(ParallelType paralle_type,
                                   std::initializer_list<Node *> producers,
                                   std::initializer_list<Node *> consumers) = 0;
};

template <typename T>
class TypeEdgeCreator : public EdgeCreator {
  virtual AbstractEdge *createEdge(ParallelType paralle_type,
                                   std::initializer_list<Node *> producers,
                                   std::initializer_list<Node *> consumers) {
    return new T(paralle_type, producers, consumers);
  }
};

std::map<EdgeType, std::shared_ptr<EdgeCreator>> &getGlobalEdgeCreatorMap();

template <typename T>
class TypeEdgeRegister {
 public:
  explicit TypeEdgeRegister(EdgeType type) {
    getGlobalEdgeCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

AbstractEdge *createEdge(ParallelType paralle_type,
                         std::initializer_list<Node *> producers,
                         std::initializer_list<Node *> consumers);

}  // namespace dag
}  // namespace nndeploy

#endif