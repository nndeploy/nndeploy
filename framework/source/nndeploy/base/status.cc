
#include "nndeploy/base/status.h"

namespace nndeploy {
namespace base {

Status::Status(int code) : code_(code) {}
Status::~Status() {}

Status::Status(const Status &other) = default;
Status &Status::operator=(const Status &other) = default;
Status &Status::operator=(const StatusCode &other) {
  code_ = other;
  return *this;
};
Status &Status::operator=(int other) {
  code_ = other;
  return *this;
};

Status::Status(Status &&other) = default;
Status &Status::operator=(Status &&other) = default;

bool Status::operator==(const Status &other) const {
  if (code_ == other.code_) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator==(const StatusCode &other) const {
  if (code_ == other) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator==(int other) const {
  if (code_ == other) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator!=(const Status &other) const {
  if (code_ != other.code_) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator!=(const StatusCode &other) const {
  if (code_ != other) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator!=(int other) const {
  if (code_ != other) {
    return true;
  } else {
    return false;
  }
};

Status::operator int() const { return code_; }

Status::operator bool() const { return code_ == kStatusCodeOk; }

Status Status::operator+(const Status &other) {
  if (code_ == kStatusCodeOk) {
    code_ = other.code_;
  }
  return *this;
}

std::string Status::desc() const {
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
