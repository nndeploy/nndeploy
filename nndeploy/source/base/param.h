#ifndef _NNDEPLOY_SOURCE_BASE_PARAM_H_
#define _NNDEPLOY_SOURCE_BASE_PARAM_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"

namespace nndeploy {
namespace base {

class Param {
 public:
  Param();
  Param(std::string name);

  ~Param();

  virtual base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

 private:
  std::string name_;
};

}  // namespace base
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_BASE_PARAM_H_ */
