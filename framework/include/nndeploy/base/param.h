#ifndef _NNDEPLOY_BASE_PARAM_H_
#define _NNDEPLOY_BASE_PARAM_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/any.h"
#include "nndeploy/base/rapidjson_include.h"

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
  virtual ~Param();

  PARAM_COPY(Param)
  PARAM_COPY_TO(Param)

  virtual base::Status set(const std::string &key, base::Any &any);

  virtual base::Status get(const std::string &key, base::Any &any);

  // 序列化：数据结构->[rapidjson::Value\stream\path\string]
  // 衍生类只需实现serialize(rapidjson::Value &json, rapidjson::Document::AllocatorType& allocator)
  virtual base::Status serialize(rapidjson::Value &json, rapidjson::Document::AllocatorType& allocator);
  virtual base::Status serialize(std::ostream &stream);
  virtual base::Status serialize(std::string &content, bool is_file);
  // 反序列化：[rapidjson::Value\stream\path\string]->数据结构
  // 衍生类只需实现deserialize(rapidjson::Value &json)
  virtual base::Status deserialize(rapidjson::Value &json);
  virtual base::Status deserialize(std::istream &stream);
  virtual base::Status deserialize(const std::string &content, bool is_file);
};

}  // namespace base
}  // namespace nndeploy

#endif /* _NNDEPLOY_BASE_PARAM_H_ */
