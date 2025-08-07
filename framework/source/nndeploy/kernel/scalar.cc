#include "nndeploy/kernel/scalar.h"

namespace nndeploy {
namespace kernel {

Scalar Scalar::operator+(const Scalar& other) const {
  if (IsFloatingPoint() || other.IsFloatingPoint()) {
    return Scalar(As<double>() + other.As<double>());
  } else if (IsSigned() || other.IsSigned()) {
    return Scalar(As<int64_t>() + other.As<int64_t>());
  } else {
    return Scalar(As<uint64_t>() + other.As<uint64_t>());
  }
}

Scalar Scalar::operator-(const Scalar& other) const {
  if (IsFloatingPoint() || other.IsFloatingPoint()) {
    return Scalar(As<double>() - other.As<double>());
  } else if (IsSigned() || other.IsSigned()) {
    return Scalar(As<int64_t>() - other.As<int64_t>());
  } else {
    return Scalar(As<uint64_t>() - other.As<uint64_t>());
  }
}

Scalar Scalar::operator*(const Scalar& other) const {
  if (IsFloatingPoint() || other.IsFloatingPoint()) {
    return Scalar(As<double>() * other.As<double>());
  } else if (IsSigned() || other.IsSigned()) {
    return Scalar(As<int64_t>() * other.As<int64_t>());
  } else {
    return Scalar(As<uint64_t>() * other.As<uint64_t>());
  }
}

Scalar Scalar::operator/(const Scalar& other) const {
  if (IsFloatingPoint() || other.IsFloatingPoint()) {
    return Scalar(As<double>() / other.As<double>());
  } else if (IsSigned() || other.IsSigned()) {
    return Scalar(As<int64_t>() / other.As<int64_t>());
  } else {
    return Scalar(As<uint64_t>() / other.As<uint64_t>());
  }
}

Scalar& Scalar::operator+=(const Scalar& other) {
  *this = *this + other;
  return *this;
}

Scalar& Scalar::operator-=(const Scalar& other) {
  *this = *this - other;
  return *this;
}

Scalar& Scalar::operator*=(const Scalar& other) {
  *this = *this * other;
  return *this;
}

Scalar& Scalar::operator/=(const Scalar& other) {
  *this = *this / other;
  return *this;
}

}  // namespace kernel
}  // namespace nndeploy