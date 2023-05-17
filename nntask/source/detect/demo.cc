#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/time_measurement.h"
#include "nntask/source/detect/task.h"

int main(int argc, char *argv[]) {
  nndeploy::base::TimeMeasurement *tm = new nndeploy::base::TimeMeasurement();

  nntask::detect::Task *task = nntask::detect::Task();

  nndeploy::inernece::InferenceParam *param =
      dynamic_cast<nndeploy::inernece::InferenceParam *>(
          task->getInferenceParam());

  task->init();

  nntask::common::Packet input;
  nntask::common::Packet output;
  task->setInput(input);
  task->setOutput(output);
  task->run();

  task->deinit();

  printf("hello world!\n");
  return 0;
}