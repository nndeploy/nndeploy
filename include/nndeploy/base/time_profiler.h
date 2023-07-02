
#ifndef _NNDEPLOY_BASE_TIME_MEASUREMENT_H_
#define _NNDEPLOY_BASE_TIME_MEASUREMENT_H_

#include "nndeploy/base/basic.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"

namespace nndeploy {
namespace base {

class NNDEPLOY_CC_API TimeMeasurement : public NonCopyable {
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