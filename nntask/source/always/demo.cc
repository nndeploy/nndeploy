#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/time_measurement.h"

int main(int argc, char *argv[]) {
  nndeploy::base::TimeMeasurement *tm = new nndeploy::base::TimeMeasurement();

  printf("hello world!\n");
  return 0;
}