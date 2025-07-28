#ifndef _NNDEPLOY_DAG_BASE_H_
#define _NNDEPLOY_DAG_BASE_H_

#include <cstring>
#include <iostream>
#include <string_view>
#include <typeindex>
#include <typeinfo>

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
namespace nndeploy {
namespace dag {

enum class NodeType {
  kNodeTypeInput = 1,         // 输入节点，无输出
  kNodeTypeOutput = 2,        // 输出节点，无输入
  kNodeTypeIntermediate = 3,  // 中间节点，既有输入也有输出
};

enum class EdgeTypeFlag {
  kBuffer = 1,
  kCvMat = 2,
  kTensor = 4,
  kParam = 8,
  kAny = 1 << 30,
  kNone = 1 << 31,
};

template <typename T>
std::string typeName() {
#if defined(__clang__)
  constexpr auto prefix = std::string_view("[T = ");
  constexpr auto suffix = "]";
  constexpr auto function = std::string_view(__PRETTY_FUNCTION__);
#elif defined(__GNUC__)
  constexpr auto prefix = std::string_view("with T = ");
  constexpr auto suffix = "]";
  constexpr auto function = std::string_view(__PRETTY_FUNCTION__);
#elif defined(_MSC_VER)
  constexpr auto prefix = std::string_view("type_name<");
  constexpr auto suffix = ">(void)";
  constexpr auto function = std::string_view(__FUNCSIG__);
#else
  return std::type_index(typeid(T)).name();
#endif

  const size_t start = function.find(prefix) + prefix.size();
  const size_t end = function.find(suffix, start);
  std::string_view type_view = function.substr(start, end - start);

  // 查找分号位置，如果存在则只返回分号前的部分
  std::string type_str(type_view);
  size_t semicolon_pos = type_str.find(';');
  if (semicolon_pos != std::string::npos) {
    return type_str.substr(0, semicolon_pos);
  }

  return type_str;
}

extern NNDEPLOY_CC_API std::string removeNamespace(
    const std::string& type_name_with_namespace);

/**
 * @brief 输入输出类型信息
 * @note 用于描述输入输出类型信息
 */
class NNDEPLOY_CC_API EdgeTypeInfo {
 public:
  EdgeTypeInfo() : type_(EdgeTypeFlag::kNone), type_name_("") {}
  ~EdgeTypeInfo() = default;

  EdgeTypeInfo(const EdgeTypeInfo& other) {
    type_ = other.type_;
    type_name_ = other.type_name_;
    type_ptr_ = other.type_ptr_;
    type_holder_ = other.type_holder_;
    edge_name_ = other.edge_name_;
  }

  EdgeTypeInfo& operator=(const EdgeTypeInfo& other) {
    if (this != &other) {
      type_ = other.type_;
      type_name_ = other.type_name_;
      type_ptr_ = other.type_ptr_;
      type_holder_ = other.type_holder_;
      edge_name_ = other.edge_name_;
    }
    return *this;
  }

  bool operator==(const EdgeTypeInfo& other) const {
    return (type_ == other.type_ && type_name_ == other.type_name_ &&
            type_ptr_ == other.type_ptr_ && edge_name_ == other.edge_name_);
  }

  bool operator!=(const EdgeTypeInfo& other) const { return !(*this == other); }

  template <typename T>
  void setType() {
    typedef typename std::decay<T>::type DT;
    if constexpr (std::is_same<DT, device::Buffer>::value) {
      type_ = EdgeTypeFlag::kBuffer;
      // type_name_ = "Buffer";
    }
#ifdef ENABLE_NNDEPLOY_OPENCV
    else if constexpr (std::is_same<DT, cv::Mat>::value) {
      type_ = EdgeTypeFlag::kCvMat;
      // type_name_ = "numpy.ndarray";
    }
#endif
    else if constexpr (std::is_same<DT, device::Tensor>::value) {
      type_ = EdgeTypeFlag::kTensor;
      // type_name_ = "Tensor";
    } else if constexpr (std::is_base_of<base::Param, DT>::value) {
      type_ = EdgeTypeFlag::kParam;
      // type_name_ = "Param";
    } else {
      type_ = EdgeTypeFlag::kAny;
      // type_name_ = std::string(typeName<DT>());
    }
    type_name_ = std::string(typeName<DT>());
    type_ptr_ = &typeid(DT);
    type_holder_ = std::make_shared<TypeHolder<DT>>();
  }

  EdgeTypeFlag getType() const { return type_; }

  void setTypeName(const std::string& type_name) {
    // NNDEPLOY_LOGI("setTypeName: %s\n", type_name.c_str());
    type_name_ = type_name;
  }
  std::string getTypeName() const { return removeNamespace(type_name_); }
  std::string getTypeNameWithNamespace() const { return type_name_; }

  std::string getUniqueTypeName() {
    // basic type
    std::string base_name = type_name_;

    // timestamp
    std::string timestamp = base::getUniqueString();

    // unique string
    std::stringstream ss;
    ss << base_name << "_" << timestamp;
    return ss.str();
  }

  const std::type_info* getTypePtr() const { return type_ptr_; }

  template <typename T>
  bool isType() const {
    return (type_ptr_ != nullptr) && (*type_ptr_ == typeid(T));
  }

  template <typename T, typename... Args>
  T* createType(Args&&... args) {
    if (!isType<T>()) {
      NNDEPLOY_LOGE("Type mismatch in createType\n");
      NNDEPLOY_LOGE(" stored=%s\n", type_ptr_->name());
      NNDEPLOY_LOGE(" requested=%s\n", typeid(T).name());
      return nullptr;
    }
    return new T(std::forward<Args>(args)...);
  }

  template <typename T>
  bool checkType() const {
    if (type_ptr_ == nullptr) {
      NNDEPLOY_LOGE("The type info is empty\n");
      NNDEPLOY_LOGE(" requested=%s\n", typeid(T).name());
      return false;
    }
    if (*type_ptr_ != typeid(T)) {
      NNDEPLOY_LOGE("The stored type mismatch\n");
      NNDEPLOY_LOGE(" stored=%s\n", type_ptr_->name());
      NNDEPLOY_LOGE(" requested=%s\n", typeid(T).name());
      return false;
    }
    return true;
  }

  void setEdgeName(const std::string& edge_name) { edge_name_ = edge_name; }
  std::string getEdgeName() const { return edge_name_; }

 public:
  // Type holder base class
  struct TypeHolderBase {
    virtual ~TypeHolderBase() = default;
  };

  // Type holder for specific type
  template <typename T>
  struct TypeHolder : TypeHolderBase {
    using Type = T;
  };

  EdgeTypeFlag type_;
  std::string type_name_;
  const std::type_info* type_ptr_{nullptr};
  std::shared_ptr<TypeHolderBase> type_holder_;
  std::string edge_name_;
};

extern NNDEPLOY_CC_API std::string nodeTypeToString(NodeType node_type);
extern NNDEPLOY_CC_API NodeType
stringToNodeType(const std::string& node_type_str);

// extern NNDEPLOY_CC_API std::string edgeTypeToString(EdgeTypeFlag edge_type);
// extern NNDEPLOY_CC_API EdgeTypeFlag
// stringToEdgeType(const std::string& edge_type_str);

struct NNDEPLOY_CC_API RunStatus {
  std::string node_name;
  bool is_running = false;
  size_t graph_run_size = 0;
  size_t run_size = 0;
  size_t completed_size = 0;
  float cost_time = -1.0f;
  float average_time = -1.0f;
  float init_time = -1.0f;

  RunStatus()
      : node_name(""),
        is_running(false),
        graph_run_size(0),
        run_size(0),
        completed_size(0),
        cost_time(-1.0f),
        average_time(-1.0f),
        init_time(-1.0f) {}
  RunStatus(const std::string& node_name, bool is_running,
            size_t graph_run_size, size_t run_size, size_t completed_size,
            float cost_time, float average_time, float init_time)
      : node_name(node_name),
        is_running(is_running),
        graph_run_size(graph_run_size),
        run_size(run_size),
        completed_size(completed_size),
        cost_time(cost_time),
        average_time(average_time),
        init_time(init_time) {}
  RunStatus(const RunStatus& other)
      : node_name(other.node_name),
        is_running(other.is_running),
        graph_run_size(other.graph_run_size),
        run_size(other.run_size),
        completed_size(other.completed_size),
        cost_time(other.cost_time),
        average_time(other.average_time),
        init_time(other.init_time) {}
  RunStatus& operator=(const RunStatus& other) {
    if (this != &other) {
      node_name = other.node_name;
      is_running = other.is_running;
      graph_run_size = other.graph_run_size;
      run_size = other.run_size;
      completed_size = other.completed_size;
      cost_time = other.cost_time;
      average_time = other.average_time;
      init_time = other.init_time;
    }
    return *this;
  }

  std::string getStatus() {
    if (is_running) {
      return "RUNNING";
    } else if (run_size > 0 && completed_size > 0 &&
               graph_run_size == completed_size) {
      return "DONE";
    } else if (run_size == 0 && completed_size == 0 &&
               std::abs(cost_time - 1.0f) < 1e-6) {
      return "INITING";
    } else if (run_size == 0 && completed_size == 0 &&
               std::abs(cost_time - 1.0f) > 1e-6) {
      return "INITED";
    } else {
      return "IDLE";
    }
  }
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_TYPE_H_ */
