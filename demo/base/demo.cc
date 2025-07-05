#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "nndeploy/base/any.h"
#include "nndeploy/framework.h"

int main(int argc, char* argv[]) {
  int ret = nndeployFrameworkInit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  nndeploy::base::Any a;
  a.construct<int>(1);
  int b = nndeploy::base::get<int>(a);
  std::cout << b << std::endl;

  nndeploy::base::Any c(std::string("hello"));
  std::string d = nndeploy::base::get<std::string>(c);
  std::cout << d << std::endl;

  // 报错
  // int e = nndeploy::base::unsafeGet<int>(c);
  // std::cout << e << std::endl;

  std::string version = nndeployGetVersion();
  std::cout << "version: " << version << std::endl;

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}