#ifndef _NNDEPLOY_KERNEL_KERNEL_H_
#define _NNDEPLOY_KERNEL_KERNEL_H_

#include "nndeploy/base/common.h"

// used for std::map
namespace std {
template <>
struct hash<nndeploy::base::DeviceType> {
  std::size_t operator()(const nndeploy::base::DeviceType& dt) const noexcept {
    return std::hash<decltype(dt.code_)>()(dt.code_) ^
           std::hash<decltype(dt.device_id_)>()(dt.device_id_);
  }
};
}  // namespace std

namespace nndeploy {
namespace kernel {

enum class UnaryKernelType : int {

  // activation kernel
  kKernelTypeElu = 0,
  kKernelTypeCelu,
  kKernelTypeRelu,
  kKernelTypeGelu,
  kKernelTypeHardSwish,
  kKernelTypeHardSigmoid,
  kKernelTypeHardShrink,
  kKernelTypeHardTanh,
  kKernelTypeLeakyRelu,
  kKernelTypeMish,
  kKernelTypeSelu,
  kKernelTypeSilu,
  kKernelTypeSoftShrink,
  kKernelTypeSoftSign,
  kKernelTypeSoftPlus,
  kKernelTypeTanh,
  kKernelTypeThreshold,
  kKernelTypeFastGelu,
  kKernelTypeQuickGelu,
  kKernelTypeSquareReLU,

  // math kernel
  kKernelTypeAbs,
  kKernelTypeAcos,
  kKernelTypeAcosh,
  kKernelTypeAsin,
  kKernelTypeAsinh,
  kKernelTypeAtan,
  kKernelTypeAtanh,
  kKernelTypeCeil,
  kKernelTypeCos,
  kKernelTypeCosh,
  kKernelTypeDigamma,
  kKernelTypeTrigamma,
  kKernelTypeErf,
  kKernelTypeErfc,
  kKernelTypeExp,
  kKernelTypeExp2,
  kKernelTypeExpm1,
  kKernelTypeFloor,
  kKernelTypeLgamma,
  kKernelTypeLog,
  kKernelTypeLog2,
  kKernelTypeLog10,
  kKernelTypeLog1p,
  kKernelTypeLogSigmoid,
  kKernelTypeNegative,
  kKernelTypeReciprocal,
  kKernelTypeReciprocalNoNan,
  kKernelTypeRint,
  kKernelTypeRound,
  kKernelTypeRsqrt,
  kKernelTypeSigmoid,
  kKernelTypeSign,
  kKernelTypeSin,
  kKernelTypeSinh,
  kKernelTypeSqrt,
  kKernelTypeSquare,
  kKernelTypeTan,
  kKernelTypeTrunc,
  kKernelTypeNotEqualZero,

  // logical kernel
  kKernelTypeLogicalNot,

  // cast kernel
  kKernelTypeCast,

  // bitwise kernel
  kKernelTypeBitwiseNot,

};

class Kernel {
 public:
  Kernel() = default;
  virtual ~Kernel() = default;
};

template <typename KernelT>
class Factory {
 public:
  using KernelType = KernelT;
  Factory() = default;
  virtual ~Factory() = default;
};

template <typename Key, typename Base, typename... Args>
class AutoFactory {
 public:
  using Creator = std::function<Base*(Args...)>;

  static AutoFactory& Instance() {
    static AutoFactory inst;
    return inst;
  }

  bool Register(Key key, Creator&& creator) {
    return creators_.emplace(key, std::move(creator)).second;
  }

  Base* New(Key key, Args... args) const {
    auto it = creators_.find(key);
    if (it == creators_.end()) return nullptr;
    return it->second(std::forward<Args>(args)...);
  }

  bool IsRegistered(Key key) const {
    return creators_.find(key) != creators_.end();
  }

 private:
  std::unordered_map<Key, Creator> creators_;
};

template <typename Key, typename Base, typename... Args>
std::unique_ptr<Base> NewObj(Key key, Args&&... args) {
  auto* raw = AutoFactory<Key, Base, Args...>::Instance().New(
      key, std::forward<Args>(args)...);
  return std::unique_ptr<Base>(raw);
}

template <typename Key, typename Base, typename... Args>
bool IsClassRegistered(Key key) {
  return AutoFactory<Key, Base, Args...>::Instance().IsRegistered(key);
}

template <typename FactoryType, typename... Args>
static std::unique_ptr<typename FactoryType::PrimitiveType> NewKernel(
    base::DeviceType device_type, Args&&... args) {
  if (!IsClassRegistered<base::DeviceType, FactoryType>(device_type)) {
    return nullptr;
  }
  auto factory = NewObj<base::DeviceType, FactoryType>(device_type);
  if (!factory) return nullptr;
  return factory->New(std::forward<Args>(args)...);
}

// kernel工厂注册
#define REGISTER_KERNEL_FACTORY(device, BaseFactory, DerivedFactory) \
  static bool g_##DerivedFactory##_reg =                             \
      ::nndeploy::kernel::AutoFactory< ::nndeploy::base::DeviceType, \
                                       BaseFactory>::Instance()      \
          .Register(device, []() { return new DerivedFactory; });

}  // namespace kernel
}  // namespace nndeploy

#endif /*  */
