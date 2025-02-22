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
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
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

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index, bool is_external) = 0;
  virtual base::Status set(cv::Mat &cv_mat, int index) = 0;
  virtual cv::Mat *create(int rows, int cols, int type, const cv::Vec3b& value,
                           int index) = 0;
  virtual bool notifyWritten(cv::Mat *cv_mat) = 0;
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

  virtual base::Status takeDataPacket(DataPacket *data_packet) = 0;
  virtual bool notifyAnyWritten(void *anything) = 0;
  virtual DataPacket *getDataPacket(const Node *node) = 0;
  virtual DataPacket *getGraphOutputDataPacket() = 0;

  virtual base::Status set(base::Param *param, int index, bool is_external) = 0;
  virtual base::Status set(base::Param &param, int index) = 0;
  template <typename T, typename... Args, typename std::enable_if<std::is_base_of<base::Param, T>::value, int>::type = 0>
  base::Param *create(int index, Args &&...args){
    DataPacket *data_packet = new DataPacket();
    if (data_packet == nullptr) {
      NNDEPLOY_LOGE("Failed to create param.\n");
      return nullptr;
    }
    base::Param *param = data_packet->create<T>(index, std::forward<Args>(args)...);
    if (param == nullptr) {
      NNDEPLOY_LOGE("Failed to create param.\n");
      return nullptr;
    }
    base::Status status = this->takeDataPacket(data_packet);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to set data packet.\n");
      delete data_packet;
      return nullptr;
    }
    return param;
  }
  virtual bool notifyWritten(base::Param *param) = 0;
  virtual base::Param *getParam(const Node *node) = 0;
  virtual base::Param *getGraphOutputParam() = 0;

  template <typename T>
  base::Status setAny(T *t, int index, bool is_external = true){
    DataPacket *data_packet = new DataPacket();
    if (data_packet == nullptr) {
      NNDEPLOY_LOGE("Failed to create any.\n");
      return base::kStatusCodeErrorOutOfMemory;
    }
    base::Status status = data_packet->setAny<T>(t, index, is_external);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to set any.\n");
      delete data_packet;
      return status;
    }
    status = this->takeDataPacket(data_packet);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to set data packet.\n");
      delete data_packet;
      return status;
    }
    return base::kStatusCodeOk;
  }
  template <typename T>
  base::Status setAny(T &t, int index){
    return this->setAny(&t, index);
  }
  template <typename T, typename... Args>
  T *createAny(int index, Args &&...args){
    DataPacket *data_packet = new DataPacket();
    if (data_packet == nullptr) {
      NNDEPLOY_LOGE("Failed to create any.\n");
      return nullptr;
    }
    T *t = data_packet->createAny<T>(index, std::forward<Args>(args)...);
    if (t == nullptr) {
      NNDEPLOY_LOGE("Failed to create any.\n");
      return nullptr;
    }
    base::Status status = this->takeDataPacket(data_packet);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to set data packet.\n");
      delete data_packet;
      return nullptr;
    }
    return t;
  }
  template <typename T>
  bool notifyAnyWritten(T *t){
    return this->notifyAnyWritten((void *)t);
  }
  template <typename T>
  T *getAny(const Node *node){
    DataPacket *data_packet = this->getDataPacket(node);
    if (data_packet == nullptr) {
      NNDEPLOY_LOGE("Failed to get data packet.\n");
      return nullptr;
    }
    T *t = data_packet->getAny<T>();
    if (t == nullptr) {
      NNDEPLOY_LOGE("Failed to get any.\n");
      return nullptr;
    }
    return t;
  }
  template <typename T>
  T *getGraphOutputAny(){
    DataPacket *data_packet = this->getGraphOutputDataPacket();
    if (data_packet == nullptr) {
      NNDEPLOY_LOGE("Failed to get data packet.\n");
      return nullptr;
    }
    T *t = data_packet->getAny<T>();
    if (t == nullptr) {
      NNDEPLOY_LOGE("Failed to get any.\n");
      return nullptr;
    }
    return t;
  }

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

  DataPacket *data_packet_ = nullptr;
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

std::map<base::EdgeType, std::shared_ptr<EdgeCreator>>
    &getGlobalEdgeCreatorMap();

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