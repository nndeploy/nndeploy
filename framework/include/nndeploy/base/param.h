#ifndef _NNDEPLOY_BASE_PARAM_H_
#define _NNDEPLOY_BASE_PARAM_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/rapidjson_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"

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

  base::Status setRequiredParams(
      const std::vector<std::string> &required_params);
  base::Status addRequiredParam(const std::string &required_param);
  base::Status removeRequiredParam(const std::string &required_param);
  base::Status clearRequiredParams();
  std::vector<std::string> getRequiredParams();

  base::Status setUiParams(const std::vector<std::string> &ui_params);
  base::Status addUiParam(const std::string &ui_param);
  base::Status removeUiParam(const std::string &ui_param);
  base::Status clearUiParams();
  std::vector<std::string> getUiParams();

  base::Status setIoParams(const std::vector<std::string> &io_params);
  base::Status addIoParam(const std::string &io_param);
  base::Status removeIoParam(const std::string &io_param);
  base::Status clearIoParams();
  std::vector<std::string> getIoParams();

  base::Status setDropdownParams(
      const std::map<std::string, std::vector<std::string>> &dropdown_params);
  base::Status addDropdownParam(
      const std::string &dropdown_param,
      const std::vector<std::string> &dropdown_values);
  base::Status removeDropdownParam(const std::string &dropdown_param);
  base::Status clearDropdownParams();
  std::map<std::string, std::vector<std::string>> getDropdownParams();

  // 序列化：数据结构->[rapidjson::Value\string\path]
  // 衍生类只需实现serialize(rapidjson::Value &json,
  // rapidjson::Document::AllocatorType& allocator)
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  virtual std::string serialize();
  virtual base::Status saveFile(const std::string &path);
  // 反序列化：[rapidjson::Value\string\path]->数据结构
  // 衍生类只需实现deserialize(rapidjson::Value &json)
  virtual base::Status deserialize(rapidjson::Value &json);
  virtual base::Status deserialize(const std::string &json_str);
  virtual base::Status loadFile(const std::string &path);

 public:
  std::vector<std::string> required_params_;
  std::vector<std::string> ui_params_;
  std::vector<std::string> io_params_;
  std::map<std::string, std::vector<std::string>> dropdown_params_;
};

extern NNDEPLOY_CC_API std::string removeJsonBrackets(
    const std::string &json_str);

extern NNDEPLOY_CC_API std::string prettyJsonStr(const std::string &json_str);

}  // namespace base
}  // namespace nndeploy

#endif /* _NNDEPLOY_BASE_PARAM_H_ */
