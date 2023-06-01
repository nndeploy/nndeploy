#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/time_measurement.h"

int main(int argc, char *argv[]) {
  nndeploy::base::TimeMeasurement *tm = new nndeploy::base::TimeMeasurement();

  tm->start("printf");
  printf("hello world!\n");
  tm->end("printf");

  tm->start("printf_1000");
  for (int i = 0; i < 1000; ++i) {
    tm->start("printf_single");
    printf("hello world!\n");
    tm->end("printf_single");
  }
  tm->end("printf_1000");

  tm->download("D:\\GitHub\\nndeploy\\build");// ??????? tm->print();

  return 0;
}