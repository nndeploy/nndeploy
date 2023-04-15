#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/time_measurement.h"

int main(int argc, char *argv[]) {
  nndeploy::base::TimeMeasurement *tm = new nndeploy::base::TimeMeasurement();

  printf("hello world!\n");
  return 0;
}