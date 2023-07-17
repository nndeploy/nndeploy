
#include "nndeploy/include/base/time_measurement.h"

namespace nndeploy {
namespace base {

enum TimePointFlag {
  kTimePointFlagStart = 0x00,
  kTimePointFlagEnd,
};

struct TimePoint {
  TimePointFlag flag_;
  std::string key_;
  std::chrono::high_resolution_clock::time_point std_tp_;
};

TimeMeasurement::TimeMeasurement() {}

TimeMeasurement::~TimeMeasurement() {}

void TimeMeasurement::start(const std::string &name) {
  if (name.empty()) {
    return;
  }
  if (name_to_start_time_.find(name) != name_to_start_time_.end()) {
    return;
  }
  name_to_start_time_[name] = std::chrono::high_resolution_clock::now();
}

void TimeMeasurement::end(const std::string &name) {
  if (name.empty()) {
    return;
  }
  if (name_to_end_time_.find(name) != name_to_end_time_.end()) {
    return;
  }
  name_to_end_time_[name] = std::chrono::high_resolution_clock::now();
}

void TimeMeasurement::download(const std::string &path) {
  // if (path.empty()) {
  //   return;
  // }
  // std::ofstream ofs(path);
  // if (!ofs.is_open()) {
  //   return;
  // }
  // for (auto &name : name_to_start_time_) {
  //   if (name_to_end_time_.find(name.first) == name_to_end_time_.end()) {
  //     continue;
  //   }
  //   auto start_time = name.second;
  //   auto end_time = name_to_end_time_[name.first];
  //   auto duration =
  //       std::chrono::duration_cast<std::chrono::microseconds>(end_time -
  //                                                             start_time)
  //           .count();
  //   ofs << name.first << " " << duration << std::endl;
  // }
  // ofs.close();
}

}  // namespace base
}  // namespace nndeploy
