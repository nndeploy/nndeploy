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
  AbstractEdge(base::ParallelType paralle_type);
  virtual ~AbstractEdge();

  virtual base::Status construct() = 0;

  virtual base::Status set(device::Buffer *buffer, int index,
                           bool is_external) = 0;
  virtual base::Status set(device::Buffer &buffer, int index) = 0;
  virtual device::Buffer *create(device::Device *device,
                                 const device::BufferDesc &desc, int index) = 0;
  virtual bool notifyWritten(device::Buffer *buffer) = 0;
  virtual device::Buffer *getBuffer(const Node *node) = 0;
  virtual device::Buffer *getGraphOutputBuffer() = 0;

  virtual base::Status set(device::Mat *mat, int index, bool is_external) = 0;
  virtual base::Status set(device::Mat &mat, int index) = 0;
  virtual device::Mat *create(device::Device *device,
                              const device::MatDesc &desc, int index,
                              const std::string &name) = 0;
  virtual bool notifyWritten(device::Mat *mat) = 0;
  virtual device::Mat *getMat(const Node *node) = 0;
  virtual device::Mat *getGraphOutputMat() = 0;

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index, bool is_external) = 0;
  virtual base::Status set(cv::Mat &cv_mat, int index) = 0;
  virtual cv::Mat *getCvMat(const Node *node) = 0;
  virtual cv::Mat *getGraphOutputCvMat() = 0;
#endif

  virtual base::Status set(device::Tensor *tensor, int index,
                           bool is_external) = 0;
  virtual base::Status set(device::Tensor &tensor, int index) = 0;
  virtual device::Tensor *create(device::Device *device,
                                 const device::TensorDesc &desc, int index,
                                 const std::string &name) = 0;
  virtual bool notifyWritten(device::Tensor *tensor) = 0;
  virtual device::Tensor *getTensor(const Node *node) = 0;
  virtual device::Tensor *getGraphOutputTensor() = 0;

  virtual base::Status set(base::Param *param, int index, bool is_external) = 0;
  virtual base::Status set(base::Param &param, int index) = 0;
  virtual base::Param *getParam(const Node *node) = 0;
  virtual base::Param *getGraphOutputParam() = 0;

  virtual base::Status set(void *anything, int index, bool is_external) = 0;
  virtual void *getAnything(const Node *node) = 0;
  virtual void *getGraphOutputAnything() = 0;

  virtual int getIndex(const Node *node) = 0;
  virtual int getGraphOutputIndex() = 0;

  virtual int getPosition(const Node *node) = 0;
  virtual int getGraphOutputPosition() = 0;

  virtual base::EdgeUpdateFlag update(const Node *node) = 0;

  virtual bool markGraphOutput();

  base::ParallelType getParallelType();

  std::vector<Node *> getProducers();
  base::Status increaseProducers(std::vector<Node *> &producers);

  std::vector<Node *> getConsumers();
  base::Status increaseConsumers(std::vector<Node *> &consumers);

  virtual bool requestTerminate() = 0;

 protected:
  bool checkNode(const Node *node);

 protected:
  base::ParallelType parallel_type_;
  std::vector<Node *> producers_;
  std::vector<Node *> consumers_;
  bool terminate_flag_ = false;
};

class EdgeCreator {
 public:
  virtual ~EdgeCreator(){};
  virtual AbstractEdge *createEdge(base::ParallelType paralle_type) = 0;
};

template <typename T>
class TypeEdgeCreator : public EdgeCreator {
  virtual AbstractEdge *createEdge(base::ParallelType paralle_type) {
    return new T(paralle_type);
  }
};

std::map<base::EdgeType, std::shared_ptr<EdgeCreator>> &
getGlobalEdgeCreatorMap();

template <typename T>
class TypeEdgeRegister {
 public:
  explicit TypeEdgeRegister(base::EdgeType type) {
    getGlobalEdgeCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

AbstractEdge *createEdge(base::ParallelType paralle_type);

AbstractEdge *recreateEdge(AbstractEdge *abstact_edge,
                           const base::ParallelType &paralle_type);

}  // namespace dag
}  // namespace nndeploy

#endif