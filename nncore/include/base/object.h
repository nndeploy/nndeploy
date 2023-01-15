/**
 * @file object.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-20
 *
 * @copyright Copyright (c) 2022
 * @note
 * # ref tvm and mnn
 */
#ifndef _NNCORE_INCLUDE_BASE_OBJECT_H_
#define _NNCORE_INCLUDE_BASE_OBJECT_H_

#include "nncore/include/base/include_c_cpp.h"
#include "nncore/include/base/macro.h"
#include "nncore/include/base/status.h"
#include "nncore/include/base/type.h"

namespace nncore {
namespace base {

class NonCopyable {
 public:
  NonCopyable() = default;
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable(const NonCopyable&&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&&) = delete;
};

}  // namespace base
}  // namespace nncore

#endif