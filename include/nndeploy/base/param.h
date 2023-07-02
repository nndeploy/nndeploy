#ifndef _NNDEPLOY_BASE_PARAM_H_
#define _NNDEPLOY_BASE_PARAM_H_

#include "nndeploy/base/basic.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"

namespace nndeploy {
namespace base {

class Param {
 public:
  Param();
  ~Param();

  virtual base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

 public:
  std::string name_;
};

}  // namespace base
}  // namespace nndeploy

#endif /* _NNDEPLOY_BASE_PARAM_H_ */
