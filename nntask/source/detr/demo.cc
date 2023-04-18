#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/time_measurement.h"

int main(int argc, char *argv[]) {
  nndeploy::TimeMeasurement tm;

  nntask::detr::InputNode *inputs = new nntask::detr::InputNode();
  nntask::detr::OutputNode *outputs = new nntask::detr::OutputNode();
  nntask::detr::Node *node = new nntask::detr::Node();

  node->init(type, path);
  node->setInput(inputs);
  node->asyncRun();

  return 0;
}