#include "nndeploy/base/param.h"

namespace nndeploy {
namespace base {

Param::Param() {}

Param::~Param() {}

base::Status Param::parse(const std::string &json, bool is_path) {
  return base::kStatusCodeOk;
}

base::Status Param::set(const std::string &key, base::Any &any) {
  return base::kStatusCodeOk;
}

base::Status Param::get(const std::string &key, base::Any &any) {
  return base::kStatusCodeOk;
}

base::Status Param::serialize(std::ostream &stream) {
  return base::kStatusCodeOk;
}

base::Status Param::deserialize(const std::string &str) {
  return base::kStatusCodeOk;
}

}  // namespace base
}  // namespace nndeploy