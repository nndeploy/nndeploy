#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"

using namespace nndeploy;

void printHelloWorld() {
  std::cout << "hello world!" << std::endl;
}

int main(int argc, char const *argv[]) {
  printHelloWorld();
  return 0;
}