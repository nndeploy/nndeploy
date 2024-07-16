
#ifndef _NNDEPLOY_NET_SESSION_H_
#define _NNDEPLOY_NET_SESSION_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/net/util.h"

namespace nndeploy {
namespace net {

class NNDEPLOY_CC_API Session : public base::NonCopyable {
 public:
  Session(){};
  virtual ~Session(){};

  virtual base::Status init(std::vector<TensorWrapper *> &tensor_repository,
                            std::vector<OpWrapper *> &op_repository,
                            std::map<std::string, device::Tensor *> &weights,
                            std::map<std::string, std::string> &weights_path_,
                            std::map<std::string, op::Op *> &weight_op_) = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status reshape(base::ShapeMap &shape_map) = 0;

  virtual base::Status preRun() = 0;
  virtual base::Status run() = 0;
  virtual base::Status postRun() = 0;
};

/**
 * @brief Session的创建类
 *
 */
class SessionCreator {
 public:
  virtual ~SessionCreator(){};

  virtual Session *createSession(base::DeviceTypeCode device_type_code,
                                 base::ParallelType parallel_type) = 0;
};

/**
 * @brief Session的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeSessionCreator : public SessionCreator {
  virtual Session *createSession(base::DeviceTypeCode device_type_code,
                                 base::ParallelType parallel_type) {
    auto Session = new T();
    return Session;
  }
};

/**
 * @brief Get the Global Session Creator Map object
 *
 * @return std::map<ExecutorType, std::map<const std::string &,
 * std::shared_ptr<SessionCreator>>>&
 */
std::map<base::DeviceTypeCode,
         std::map<base::ParallelType, std::shared_ptr<SessionCreator>>> &
getGlobalSessionCreatorMap();

/**
 * @brief Session的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeSessionRegister {
 public:
  explicit TypeSessionRegister(base::DeviceTypeCode device_type_code,
                               base::ParallelType parallel_type) {
    getGlobalSessionCreatorMap()[device_type_code][Session_type] =
        std::shared_ptr<T>(new T());
  }
};

Session *createSession(base::DeviceType device_type,
                       base::ParallelType parallel_type);

}  // namespace net
}  // namespace nndeploy

#endif
