
#ifndef _NNDEPLOY_NET_SESSION_H_
#define _NNDEPLOY_NET_SESSION_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/any.h"
#include "nndeploy/net/tensor_pool.h"
#include "nndeploy/net/util.h"

namespace nndeploy {
namespace net {

class NNDEPLOY_CC_API Session : public base::NonCopyable {
 public:
  Session(const base::DeviceType &device_type) : device_type_(device_type) {};
  virtual ~Session() {};

  virtual base::Status init(
      std::vector<TensorWrapper *> &tensor_repository,
      std::vector<OpWrapper *> &op_repository, bool is_dynamic_shape,
      base::ShapeMap max_shape,
      TensorPoolType tensor_pool_type =
          kTensorPool1DSharedObjectTypeGreedyBySizeImprove) = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status reshape(base::ShapeMap &shape_map) = 0;

  /**
   * @brief 获取推理所需的内存大小
   *
   * @return int64_t
   */
  virtual int64_t getMemorySize();
  /**
   * @brief 设置推理所需的内存（推理内存由外部分配）
   *
   * @param buffer
   * @return base::Status
   */
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::Status preRun() = 0;
  virtual base::Status run() = 0;
  virtual base::Status postRun() = 0;

 protected:
  base::DeviceType device_type_;
  TensorPoolType tensor_pool_type_ =
      kTensorPool1DSharedObjectTypeGreedyBySizeImprove;
  TensorPool *tensor_pool_;
  bool is_dynamic_shape_ = false;                // 是否是动态shape
  base::ShapeMap max_shape_ = base::ShapeMap();  // 当为动态输入时最大shape
  std::vector<TensorWrapper *> tensor_repository_;
  std::vector<OpWrapper *> op_repository_;
};

/**
 * @brief Session的创建类
 *
 */
class SessionCreator {
 public:
  virtual ~SessionCreator() {};

  virtual Session *createSession(const base::DeviceType &device_type,
                                 base::ParallelType parallel_type) = 0;
};

/**
 * @brief Session的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeSessionCreator : public SessionCreator {
  virtual Session *createSession(const base::DeviceType &device_type,
                                 base::ParallelType parallel_type) {
    auto Session = new T(device_type);
    return Session;
  }
};

/**
 * @brief Get the Global Session Creator Map object
 *
 * @return std::map<ExecutorType, std::map<const std::string &,
 * std::shared_ptr<SessionCreator>>>&
 */
std::map<base::ParallelType, std::shared_ptr<SessionCreator>> &
getGlobalSessionCreatorMap();

/**
 * @brief Session的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeSessionRegister {
 public:
  explicit TypeSessionRegister(base::ParallelType parallel_type) {
    getGlobalSessionCreatorMap()[parallel_type] = std::shared_ptr<T>(new T());
  }
};

Session *createSession(const base::DeviceType &device_type,
                       base::ParallelType parallel_type);

}  // namespace net
}  // namespace nndeploy

#endif
