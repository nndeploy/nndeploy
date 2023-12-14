
#include "nndeploy/base/status.h"

namespace nndeploy {
namespace base {

Status::Status(int code) : code_(code) {}
Status::~Status() {}

Status &Status::operator=(int code) {
  code_ = code;
  return *this;
};

bool Status::operator==(int code) {
  if (code_ == code) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator!=(int code) {
  if (code_ != code) {
    return true;
  } else {
    return false;
  }
};

Status::operator int() { return code_; }

Status::operator bool() { return code_ == kStatusCodeOk; }

std::string Status::desc() {
  std::string str;
  switch (code_) {
    default:
      str = std::to_string(static_cast<int>(code_));
      break;
  }
  return str;
};

}  // namespace base
}  // namespace nndeploy
