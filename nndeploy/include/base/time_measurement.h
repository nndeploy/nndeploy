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
#ifndef _NNDEPLOY_INCLUDE_BASE_TIME_MEASUREMENT_H_
#define _NNDEPLOY_INCLUDE_BASE_TIME_MEASUREMENT_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"

namespace nndeploy {
namespace base {

class TimeMeasurement : public NonCopyable {
 public:
  TimeMeasurement();
  virtual ~TimeMeasurement();

  void start(const std::string &name);
  void end(const std::string &name);

  void download(const std::string &path);
 private:
  std::map<std::string, std::chrono::high_resolution_clock::time_point>
      name_to_start_time_;
  std::map<std::string, std::chrono::high_resolution_clock::time_point>
      name_to_end_time_;
};

}  // namespace base
}  // namespace nndeploy

#endif