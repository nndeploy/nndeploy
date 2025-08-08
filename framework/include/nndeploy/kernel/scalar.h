#ifndef _NNDEPLOY_KERNEL_SCALAR_H_
#define _NNDEPLOY_KERNEL_SCALAR_H_

// Kernel中只传递裸数据、裸指针
// Scalar是一个任意类型标量的包装类

#include <complex>
#include <cstdint>
#include <type_traits>

namespace nndeploy {
namespace kernel {

  
class Scalar {
 public:
  // Default constructor
  Scalar() : Scalar(int32_t(0)) {}

  // Constructor for complex types
  template <typename T, typename std::enable_if<
                            std::is_same<std::complex<float>, T>::value ||
                                std::is_same<std::complex<double>, T>::value,
                            int>::type = 0>
  Scalar(const T& value) : active_tag_(HAS_C) {
    value_.c.real = value.real();
    value_.c.imag = value.imag();
  }

  // Constructor for bool type
  template <typename T, typename std::enable_if<std::is_same<T, bool>::value,
                                                int>::type = 0>
  Scalar(const T& value) : active_tag_(HAS_B) {
    value_.b = value;
  }

  // Constructor for signed integral types
  template <typename T, typename std::enable_if<std::is_integral<T>::value &&
                                                    std::is_signed<T>::value,
                                                int>::type = 0>
  Scalar(const T& value) : active_tag_(HAS_S) {
    value_.s = value;
  }

  // Constructor for unsigned integral types
  template <typename T,
            typename std::enable_if<std::is_integral<T>::value &&
                                        std::is_unsigned<T>::value &&
                                        !std::is_same<T, bool>::value,
                                    int>::type = 0>
  Scalar(const T& value) : active_tag_(HAS_U) {
    value_.u = value;
  }

  // Constructor for floating point types
  template <typename T, typename std::enable_if<
                            std::is_floating_point<T>::value, int>::type = 0>
  Scalar(const T& value) : active_tag_(HAS_D) {
    value_.d = value;
  }

  // Assignment operator
  template <typename T, typename std::enable_if<!std::is_same<T, Scalar>::value,
                                                int>::type = 0>
  Scalar& operator=(const T& value) {
    *this = Scalar(value);
    return *this;
  }

  // Copy assignment operator
  Scalar& operator=(const Scalar& other) {
    value_ = other.value_;
    active_tag_ = other.active_tag_;
    return *this;
  }

  // Conversion functions
  template <typename T,
            typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
  T As() const {
    switch (active_tag_) {
      case HAS_B:
        return static_cast<T>(value_.b);
      case HAS_S:
        return static_cast<T>(value_.s);
      case HAS_U:
        return static_cast<T>(value_.u);
      case HAS_D:
        return static_cast<T>(value_.d);
      default:
        throw std::runtime_error("Invalid active tag");
    }
  }

  template <typename T,
            typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
  T Value() const {
    return As<T>();
  }

  template <typename T, typename std::enable_if<
                            std::is_same<std::complex<float>, T>::value ||
                                std::is_same<std::complex<double>, T>::value,
                            int>::type = 0>
  T Value() const {
    if (!IsComplex()) {
      return T(As<double>(), 0.0);
    }
    return T(value_.c.real, value_.c.imag);
  }

  // Type checking functions
  bool IsBool() const { return active_tag_ == HAS_B; }
  bool IsIntegral() const {
    return active_tag_ == HAS_S || active_tag_ == HAS_U;
  }
  bool IsFloatingPoint() const { return active_tag_ == HAS_D; }
  bool IsSigned() const { return active_tag_ == HAS_S || active_tag_ == HAS_D; }
  bool IsUnsigned() const { return active_tag_ == HAS_U; }
  bool IsComplex() const { return active_tag_ == HAS_C; }

  // Arithmetic operations
  Scalar operator+(const Scalar& other) const;
  Scalar operator-(const Scalar& other) const;
  Scalar operator*(const Scalar& other) const;
  Scalar operator/(const Scalar& other) const;

  Scalar& operator+=(const Scalar& other);
  Scalar& operator-=(const Scalar& other);
  Scalar& operator*=(const Scalar& other);
  Scalar& operator/=(const Scalar& other);

 private:
  union Value {
    bool b;
    int64_t s;
    uint64_t u;
    double d;
    struct {
      double real;
      double imag;
    } c;
  } value_;
  enum { HAS_B, HAS_S, HAS_U, HAS_D, HAS_C, HAS_NONE } active_tag_;
};

}  // namespace kernel
}  // namespace nndeploy

#endif  // _NNDEPLOY_KERNEL_SCALAR_H_