
#include "nndeploy/include/base/status.h"

namespace nndeploy {
namespace base {

Status& Status::operator=(int32_t code) {
  code_ = code;
  return *this;
};

bool Status::operator==(int32_t code_) { return code_ == code_; };

bool Status::operator!=(int32_t code_) { return code_ != code_; };

Status::operator int32_t() { return code_; }

Status::operator bool() { return code_ == kStatusCodeOk; }

std::string Status::desc() {
  std::ostream out;
  out << code_;
  return out.str();
};

}  // namespace base
}  // namespace nndeploy
