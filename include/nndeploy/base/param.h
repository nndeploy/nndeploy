#ifndef _NNDEPLOY_BASE_PARAM_H_
#define _NNDEPLOY_BASE_PARAM_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"

namespace nndeploy {
namespace base {

#define PARAM_COPY(param_type)                                       \
 public:                                                             \
  virtual std::shared_ptr<nndeploy::base::Param> copy() {            \
    std::shared_ptr<nndeploy::base::Param> param(new param_type());  \
    param_type *param_ptr = dynamic_cast<param_type *>(param.get()); \
    *param_ptr = *this;                                              \
    return param;                                                    \
  }

#define PARAM_COPY_TO(param_type)                                       \
 public:                                                                \
  virtual nndeploy::base::Status copyTo(nndeploy::base::Param *param) { \
    param_type *param_ptr = dynamic_cast<param_type *>(param);          \
    if (nullptr == param_ptr) {                                         \
      NNDEPLOY_LOGE("dynamic cast to %s failed\n", #param_type);        \
      return nndeploy::base::kStatusCodeErrorInvalidParam;              \
    }                                                                   \
    *param_ptr = *this;                                                 \
    return nndeploy::base::kStatusCodeOk;                               \
  }

class NNDEPLOY_CC_API Param {
 public:
  Param();
  ~Param();

  Param(const Param &param) = default;
  Param &operator=(const Param &param) = default;

  PARAM_COPY(Param)
  PARAM_COPY_TO(Param)

  void setName(const std::string &name);
  std::string getName();

  virtual base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

 public:
  std::string name_;
};

}  // namespace base
}  // namespace nndeploy

#endif /* _NNDEPLOY_BASE_PARAM_H_ */
