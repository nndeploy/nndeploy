/**
 * @file profiler.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-21
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNCORE_INCLUDE_BASE_PROFILER_H_
#define _NNCORE_INCLUDE_BASE_PROFILER_H_

#include "nncore/include/base/include_c_cpp.h"
#include "nncore/include/base/macro.h"
#include "nncore/include/base/object.h"
#include "nncore/include/base/type.h"


namespace nncore {
namespace base {

class Profiler : public NonCopyable {
 public:
  Profiler();
  virtual ~Profiler();

  void start(const std::string &name);
  void end(const std::string &name);

  void download(const std::string &path);
};

}  // namespace base
}  // namespace nncore

#endif