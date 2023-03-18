
#include "nndeploy/include/base/time_measurement.h"

#include <fstream>
#include <sstream>

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"

namespace nndeploy {
namespace base {

TimeMeasurement::TimeMeasurement() {}
TimeMeasurement::~TimeMeasurement() {}

void TimeMeasurement::start(const std::string& name) {
  clock_t start = clock();

  auto iter = time_record.find(name);
  if (iter != time_record.end()) {
    iter->second->start_ = start;
    record_check.push(iter->second);
  } else {
    time_record_seperate* name_now = new time_record_seperate(name, start);
    time_record[name] = name_now;
    record_check.push(name_now);
  }
}

void TimeMeasurement::end(
    const std::string& name) {  //这里其实不需要输入参数也可以
  clock_t end = clock();
  double duiration_now = (double)(end - record_check.top()->start_) / CLK_TCK;
  record_check.top()->count_++;
  record_check.top()->duration_sum_ =
      record_check.top()->duration_sum_ + duiration_now;
  record_check.pop();
}

void TimeMeasurement::download(const std::string& path) {
  std::ofstream outFile;
  std::string time_path = path + "/time_measurement" +
      std::to_string((double)(time_record.begin()->second->start_)) + ".csv";
  outFile.open(time_path, std::ios::out);

  outFile << "func_name"
          << ","
          << "time_sum"
          << ","
          << "counts"
          << ","
          << "time_average" << std::endl;

  std::string name;
  double time_sum, time_avg;
  int count;

  for (auto iter : time_record) {
    std::string name = iter.second->name_;
    time_sum = iter.second->duration_sum_;
    count = iter.second->count_;
    time_avg = (double)(iter.second->duration_sum_ / iter.second->count_);

    outFile << name << "," << time_sum << "," << count << "," << time_avg
            << std::endl;
  }

  outFile.close();
}
}  // namespace base

}  // namespace nndeploy
