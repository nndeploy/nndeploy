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
#ifndef _NNDEPLOY_INCLUDE_BASE_PROFILER_H_
#define _NNDEPLOY_INCLUDE_BASE_PROFILER_H_

#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/type.h"

namespace nndeploy {
namespace base {

class Profiler {
  public:
    void start(const std::string &name);
    void end(const std::string &name);

    void download(const std::string &path);
};

}
}  // namespace nndeploy

#endif