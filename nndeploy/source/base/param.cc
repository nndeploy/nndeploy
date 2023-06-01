#include "nndeploy/source/base/param.h"

namespace nndeploy {
namespace base {

Param::Param() {}
Param::Param(std::string name) : name_(name) {}

Param::~Param() {}

base::Status Param::parse(const std::string &json, bool is_path) {
  return base::kStatusCodeOk;
}

base::Status Param::set(const std::string &key, base::Value &value) {
  return base::kStatusCodeOk;
}

base::Status Param::get(const std::string &key, base::Value &value) {
  return base::kStatusCodeOk;
}

}  // namespace base
}  // namespace nndeploy