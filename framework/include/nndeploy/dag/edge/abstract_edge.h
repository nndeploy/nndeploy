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

class NNDEPLOY_CC_API AbstractEdge : public base::NonCopyable {
 public:
  AbstractEdge(base::ParallelType paralle_type);
  virtual ~AbstractEdge();

  virtual base::Status setQueueMaxSize(int queue_max_size) = 0;
  virtual base::Status setQueueOverflowPolicy(base::QueueOverflowPolicy policy,
                                              int drop_count);

  virtual bool empty() = 0;

  virtual base::Status construct() = 0;

  virtual base::Status set(device::Buffer *buffer, bool is_external) = 0;
  virtual device::Buffer *create(device::Device *device,
                                 const device::BufferDesc &desc) = 0;
  virtual bool notifyWritten(device::Buffer *buffer) = 0;
  virtual device::Buffer *getBuffer(const Node *node) = 0;
  virtual device::Buffer *getGraphOutputBuffer() = 0;

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, bool is_external) = 0;
  virtual cv::Mat *create(int rows, int cols, int type,
                          const cv::Vec3b &value) = 0;
  virtual bool notifyWritten(cv::Mat *cv_mat) = 0;
  virtual cv::Mat *getCvMat(const Node *node) = 0;
  virtual cv::Mat *getGraphOutputCvMat() = 0;
#endif

  virtual base::Status set(device::Tensor *tensor, bool is_external) = 0;
  virtual device::Tensor *create(device::Device *device,
                                 const device::TensorDesc &desc,
                                 const std::string &name) = 0;
  virtual bool notifyWritten(device::Tensor *tensor) = 0;
  virtual device::Tensor *getTensor(const Node *node) = 0;
  virtual device::Tensor *getGraphOutputTensor() = 0;

  virtual base::Status takeDataPacket(DataPacket *data_packet) = 0;
  virtual bool notifyWritten(void *anything) = 0;
  virtual DataPacket *getDataPacket(const Node *node) = 0;
  virtual DataPacket *getGraphOutputDataPacket() = 0;

  virtual base::Status set(base::Param *param, bool is_external) = 0;
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<base::Param, T>::value,
                                    int>::type = 0>
  base::Param *create(Args &&...args) {
    DataPacket *data_packet = new DataPacket();
    if (data_packet == nullptr) {
      NNDEPLOY_LOGE("Failed to create param.\n");
      return nullptr;
    }
    this->increaseIndex();
    data_packet->setIndex(index_);
    base::Param *param = data_packet->create<T>(std::forward<Args>(args)...);
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

  virtual void *getGraphOutputPtr() = 0;

  template <typename T>
  base::Status set(T *t, bool is_external = true) {
    DataPacket *data_packet = new DataPacket();
    if (data_packet == nullptr) {
      NNDEPLOY_LOGE("Failed to create any.\n");
      return base::kStatusCodeErrorOutOfMemory;
    }
    this->increaseIndex();
    data_packet->setIndex(index_);
    base::Status status = data_packet->set<T>(t, is_external);
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
    // bool is_notify = this->notifyWritten(t);
    // if (!is_notify) {
    //   NNDEPLOY_LOGE("Failed to notify written.\n");
    //   return base::kStatusCodeErrorInvalidParam;
    // }
    return base::kStatusCodeOk;
  }
  template <typename T, typename... Args>
  T *create(Args &&...args) {
    DataPacket *data_packet = new DataPacket();
    if (data_packet == nullptr) {
      NNDEPLOY_LOGE("Failed to create any.\n");
      return nullptr;
    }
    this->increaseIndex();
    data_packet->setIndex(index_);
    T *t = data_packet->create<T>(std::forward<Args>(args)...);
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
  bool notifyWritten(T *t) {
    return this->notifyWritten((void *)t);
  }
  template <typename T>
  T *get(const Node *node) {
    DataPacket *data_packet = this->getDataPacket(node);
    if (data_packet == nullptr) {
      NNDEPLOY_LOGE("Failed to get data packet.\n");
      return nullptr;
    }
    if (parallel_type_ == base::kParallelTypePipeline) {
      Node *tmp_node = const_cast<Node *>(node);
      if (std::find(producers_.begin(), producers_.end(), tmp_node) !=
          producers_.end()) {
        T *t = ((PipelineDataPacket *)data_packet)->getDirect<T>();
        if (t == nullptr) {
          NNDEPLOY_LOGE("Failed to get any.\n");
          return nullptr;
        }
        return t;
      } else {
        T *t = ((PipelineDataPacket *)data_packet)->get<T>();
        if (t == nullptr) {
          NNDEPLOY_LOGE("Failed to get any.\n");
          return nullptr;
        }
        return t;
      }
    } else {
      T *t = data_packet->get<T>();
      if (t == nullptr) {
        NNDEPLOY_LOGE("Failed to get any.\n");
        return nullptr;
      }
      return t;
    }
  }
  template <typename T>
  T *getGraphOutput() {
    DataPacket *data_packet = this->getGraphOutputDataPacket();
    if (data_packet == nullptr) {
      NNDEPLOY_LOGE("Failed to get data packet.\n");
      return nullptr;
    }
    if (parallel_type_ == base::kParallelTypePipeline) {
      T *t = ((PipelineDataPacket *)data_packet)->get<T>();
      if (t == nullptr) {
        NNDEPLOY_LOGE("Failed to get any.\n");
        return nullptr;
      }
      return t;
    } else {
      T *t = data_packet->get<T>();
      if (t == nullptr) {
        NNDEPLOY_LOGE("Failed to get any.\n");
        return nullptr;
      }
      return t;
    }
  }

  template <typename PY_WRAPPER, typename T>
  base::Status set4py(PY_WRAPPER *wrapper, T *t, bool is_external = true) {
    DataPacket *data_packet = new DataPacket();
    if (data_packet == nullptr) {
      NNDEPLOY_LOGE("Failed to create any.\n");
      return base::kStatusCodeErrorOutOfMemory;
    }
    this->increaseIndex();
    data_packet->setIndex(index_);
    base::Status status =
        data_packet->set4py<PY_WRAPPER, T>(wrapper, t, is_external);
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
    // bool is_notify = this->notifyWritten(t);
    // if (!is_notify) {
    //   NNDEPLOY_LOGE("Failed to notify written.\n");
    //   return base::kStatusCodeErrorInvalidParam;
    // }
    return base::kStatusCodeOk;
  }

  virtual int64_t getIndex(const Node *node) = 0;
  virtual int64_t getGraphOutputIndex() = 0;
  virtual void resetIndex();

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
  void increaseIndex();

 protected:
  base::ParallelType parallel_type_;
  std::vector<Node *> producers_;
  std::vector<Node *> consumers_;
  bool terminate_flag_ = false;
  int64_t index_ = -1;
  DataPacket *data_packet_ = nullptr;
};

class EdgeCreator {
 public:
  virtual ~EdgeCreator() {};
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