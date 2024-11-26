/*!
 * \file Any.h
 * \brief ref dmlc-core/include/dmlc/Any.h
 */

#ifndef _NNDEPLOY_BASE_Any_H_
#define _NNDEPLOY_BASE_Any_H_

// This code need c++11 to compile
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"

namespace nndeploy {
namespace base {

// forward declare Any;
class Any;

/*!
 * Get a reference to content stored in the Any as type T.
 * This will cause an error if
 * T does not match the type stored.
 * This function is not part of std::Any standard.
 *
 * \param src The source source Any container.
 * \return The reference of content
 * \tparam T The type of the value to be fetched.
 */
template <typename T>
inline T& get(Any& src);  // NOLINT(*)

/*!
 * Get the const reference content stored in the Any as type T.
 * This will cause an error if
 * T does not match the type stored.
 * This function is not part of std::Any standard.
 *
 * \param src The source source Any container.
 * \return The reference of content
 * \tparam T The type of the value to be fetched.
 */
template <typename T>
inline const T& get(const Any& src);

/*!
 * The "unsafe" versions of get. It is required when where we know
 * what type is stored in the Any and can't use typeid() comparison,
 * e.g., when our types may travel across different shared libraries.
 * This function is not part of std::Any standard.
 *
 * \param src The source source Any container.
 * \return The reference of content
 * \tparam T The type of the value to be fetched.
 */
template <typename T>
inline T& unsafeGet(Any& src);  // NOLINT(*)

/*!
 * The "unsafe" versions of get. It is required when where we know
 * what type is stored in the Any and can't use typeid() comparison,
 * e.g., when our types may travel across different shared libraries.
 * This function is not part of std::Any standard.
 *
 * \param src The source source Any container.
 * \return The reference of content
 * \tparam T The type of the value to be fetched.
 */
template <typename T>
inline const T& unsafeGet(const Any& src);

/*!
 * \brief An Any class that is compatible to std::Any in c++17.
 *
 * \code
 *   base::Any a = std::string("mydear"), b = 1;
 *   // get reference out and add it
 *   base::get<int>(b) += 1;
 *   // a is now string
 *   LOG(INFO) << base::get<std::string>(a);
 *   // a is now 2, the string stored will be properly destructed
 *   a = std::move(b);
 *   LOG(INFO) << base::get<int>(a);
 * \endcode
 * \sa get
 */
class Any {
 public:
  /*! \brief default constructor */
  inline Any() = default;
  /*!
   * \brief move constructor from another Any
   * \param other The other Any to be moved
   */
  inline Any(Any&& other);  // NOLINT(*)

  /*!
   * \brief copy constructor
   * \param other The other Any to be copied
   */
  inline Any(const Any& other);  // NOLINT(*)

  /*!
   * \brief constructor from Any types
   * \param other The other types to be constructed into Any.
   * \tparam T The value type of other.
   */
  template <typename T>
  inline Any(T&& other);  // NOLINT(*)

  /*! \brief destructor */
  inline ~Any();

  /*!
   * \brief assign operator from other
   * \param other The other Any to be copy or moved.
   * \return self
   */
  inline Any& operator=(Any&& other);

  /*!
   * \brief assign operator from other
   * \param other The other Any to be copy or moved.
   * \return self
   */
  inline Any& operator=(const Any& other);

  /*!
   * \brief assign operator from Any type.
   * \param other The other Any to be copy or moved.
   * \tparam T The value type of other.
   * \return self
   */
  template <typename T>
  inline Any& operator=(T&& other);

  /*!
   * \return whether the container is empty.
   */
  inline bool empty() const;

  /*!
   * \brief clear the content of container
   */
  inline void clear();

  /*!
   * swap current content with other
   * \param other The other data to be swapped.
   */
  inline void swap(Any& other);  // NOLINT(*)

  /*!
   * \return The type_info about the stored type.
   */
  inline const std::type_info& type() const;

  /*! \brief Construct value of type T inplace */
  template <typename T, typename... Args>
  inline void construct(Args&&... args);

 private:
  //! \cond Doxygen_Suppress
  // declare of helper class
  template <typename T>
  class TypeOnHeap;

  template <typename T>
  class TypeOnStack;

  template <typename T>
  class TypeInfo;

  // size of stack space, it takes 32 bytes for one Any type.
  static const size_t kStack = sizeof(void*) * 3;
  static const size_t kAlign = sizeof(void*);

  // container use dynamic storage only when space runs lager
  union Data {
    // stack space
    std::aligned_storage<kStack, kAlign>::type stack;
    // pointer to heap space
    void* pheap;
  };

  // type specific information
  struct Type {
    // destructor function
    void (*destroy)(Data* data);
    // copy constructor
    void (*createFromData)(Data* dst, const Data& src);
    // the type info function
    const std::type_info* ptype_info;
  };

  // constant to check if data can be stored on heap.
  template <typename T>
  struct data_on_stack {
    static const bool value = alignof(T) <= kAlign && sizeof(T) <= kStack;
  };

  // declare friend with
  template <typename T>
  friend T& get(Any& src);  // NOLINT(*)

  template <typename T>
  friend const T& get(const Any& src);

  template <typename T>
  friend T& unsafeGet(Any& src);  // NOLINT(*)

  template <typename T>
  friend const T& unsafeGet(const Any& src);

  // internal construct function
  inline void construct(Any&& other);

  // internal construct function
  inline void construct(const Any& other);

  // internal function to check if type is correct.
  template <typename T>
  inline void checkType() const;

  template <typename T>
  inline void checkTypeByName() const;

  // internal type specific information
  const Type* type_{nullptr};

  // internal data
  Data data_;
};

template <typename T>
inline Any::Any(T&& other) {
  // 将 T 转换为其基础类型，去掉引用和常量修饰符
  typedef typename std::decay<T>::type DT;
  // 检查 T 是否为 Any 类型
  if (std::is_same<DT, Any>::value) {
    // 如果是，调用内部构造函数
    this->construct(std::forward<T>(other));
  } else {
    // 确保 T 是可拷贝构造的
    static_assert(std::is_copy_constructible<DT>::value,
                  "Any can only hold value that is copy constructable");
    // 获取类型信息并存储
    type_ = TypeInfo<DT>::getType();
    // 检查是否可以在栈上存储
    if (data_on_stack<DT>::value) {
#pragma GCC diagnostic push
#if 6 <= __GNUC__
#pragma GCC diagnostic ignored "-Wplacement-new"
#endif
      new (&(data_.stack)) DT(std::forward<T>(other));  // 在栈上构造对象
#pragma GCC diagnostic pop
    } else {
      data_.pheap = new DT(std::forward<T>(other));  // 否则在堆上构造对象
    }
  }
}

inline Any::Any(Any&& other) { this->construct(std::move(other)); }

inline Any::Any(const Any& other) { this->construct(other); }

inline void Any::construct(Any&& other) {
  type_ = other.type_;
  data_ = other.data_;
  other.type_ = nullptr;
}

inline void Any::construct(const Any& other) {
  type_ = other.type_;
  if (type_ != nullptr) {
    type_->createFromData(&data_, other.data_);
  }
}

template <typename T, typename... Args>
inline void Any::construct(Args&&... args) {
  clear();
  typedef typename std::decay<T>::type DT;
  type_ = TypeInfo<DT>::getType();
  if (data_on_stack<DT>::value) {
#pragma GCC diagnostic push
#if 6 <= __GNUC__
#pragma GCC diagnostic ignored "-Wplacement-new"
#endif
    new (&(data_.stack)) DT(std::forward<Args>(args)...);
#pragma GCC diagnostic pop
  } else {
    data_.pheap = new DT(std::forward<Args>(args)...);
  }
}

inline Any::~Any() { this->clear(); }

inline Any& Any::operator=(Any&& other) {
  Any(std::move(other)).swap(*this);
  return *this;
}

inline Any& Any::operator=(const Any& other) {
  Any(other).swap(*this);
  return *this;
}

template <typename T>
inline Any& Any::operator=(T&& other) {
  Any(std::forward<T>(other)).swap(*this);
  return *this;
}

inline void Any::swap(Any& other) {  // NOLINT(*)
  std::swap(type_, other.type_);
  std::swap(data_, other.data_);
}

inline void Any::clear() {
  if (type_ != nullptr) {
    if (type_->destroy != nullptr) {
      type_->destroy(&data_);
    }
    type_ = nullptr;
  }
}

inline bool Any::empty() const { return type_ == nullptr; }

inline const std::type_info& Any::type() const {
  if (type_ != nullptr) {
    return *(type_->ptype_info);
  } else {
    return typeid(void);
  }
}

template <typename T>
inline void Any::checkType() const {
  if (type_ == nullptr) {
    NNDEPLOY_LOGE("The Any container is empty\n");
    NNDEPLOY_LOGE(" requested=%s\n", typeid(T).name());
    assert(false);
  }
  if (*(type_->ptype_info) != typeid(T)) {
    NNDEPLOY_LOGE("The stored type mismatch\n");
    NNDEPLOY_LOGE(" stored=%s\n", type_->ptype_info->name());
    NNDEPLOY_LOGE(" requested=%s\n", typeid(T).name());
    assert(false);
  }
}

template <typename T>
inline void Any::checkTypeByName() const {
  if (type_ == nullptr) {
    NNDEPLOY_LOGE("The Any container is empty\n");
    NNDEPLOY_LOGE(" requested=%s\n", typeid(T).name());
    assert(false);
  }
  if (strcmp(type_->ptype_info->name(), typeid(T).name()) != 0) {
    NNDEPLOY_LOGE("The stored type name mismatch\n");
    NNDEPLOY_LOGE(" stored=%s\n", type_->ptype_info->name());
    NNDEPLOY_LOGE(" requested=%s\n", typeid(T).name());
    assert(false);
  }
}

template <typename T>
inline const T& get(const Any& src) {
  src.checkType<T>();
  return *Any::TypeInfo<T>::getPtr(&(src.data_));
}

template <typename T>
inline T& get(Any& src) {  // NOLINT(*)
  src.checkType<T>();
  return *Any::TypeInfo<T>::getPtr(&(src.data_));
}

template <typename T>
inline const T& unsafeGet(const Any& src) {
  src.checkTypeByName<T>();
  return *Any::TypeInfo<T>::getPtr(&(src.data_));
}

template <typename T>
inline T& unsafeGet(Any& src) {  // NOLINT(*)
  src.checkTypeByName<T>();
  return *Any::TypeInfo<T>::getPtr(&(src.data_));
}

template <typename T>
class Any::TypeOnHeap {
 public:
  inline static T* getPtr(Any::Data* data) {
    return static_cast<T*>(data->pheap);
  }
  inline static const T* getPtr(const Any::Data* data) {
    return static_cast<const T*>(data->pheap);
  }
  inline static void createFromData(Any::Data* dst, const Any::Data& data) {
    dst->pheap = new T(*getPtr(&data));
  }
  inline static void destroy(Data* data) {
    delete static_cast<T*>(data->pheap);
  }
};

template <typename T>
class Any::TypeOnStack {
 public:
  inline static T* getPtr(Any::Data* data) {
    return reinterpret_cast<T*>(&(data->stack));
  }
  inline static const T* getPtr(const Any::Data* data) {
    return reinterpret_cast<const T*>(&(data->stack));
  }
  inline static void createFromData(Any::Data* dst, const Any::Data& data) {
    new (&(dst->stack)) T(*getPtr(&data));
  }
  inline static void destroy(Data* data) {
    T* dptr = reinterpret_cast<T*>(&(data->stack));
    dptr->~T();
  }
};

template <typename T>
class Any::TypeInfo
    : public std::conditional<Any::data_on_stack<T>::value, Any::TypeOnStack<T>,
                              Any::TypeOnHeap<T> >::type {
 public:
  inline static const Type* getType() {
    static TypeInfo<T> tp;
    return &(tp.type_);
  }

 private:
  // local type
  Type type_;
  // constructor
  TypeInfo() {
    if (std::is_pod<T>::value && data_on_stack<T>::value) {
      type_.destroy = nullptr;
    } else {
      type_.destroy = TypeInfo<T>::destroy;
    }
    type_.createFromData = TypeInfo<T>::createFromData;
    type_.ptype_info = &typeid(T);
  }
};

}  // namespace base
}  // namespace nndeploy

#endif