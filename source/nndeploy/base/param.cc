#include "nndeploy/base/param.h"

namespace nndeploy {
namespace base {

Param::Param() {}

Param::~Param() {}

Param::Param(const Param &param) { name_ = param.name_; }
Param &Param::operator=(const Param &param) {
  name_ = param.name_;
  return *this;
}

void Param::setName(const std::string &name) { name_ = name; }
std::string Param::getName() { return name_; }

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