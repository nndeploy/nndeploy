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

  // 测试传入一个智能指针
  std::shared_ptr<int> ptr = std::make_shared<int>(42);
  nndeploy::base::Any smart_ptr_any;
  smart_ptr_any.construct<std::shared_ptr<int>>(ptr);
  std::shared_ptr<int> retrieved_ptr = nndeploy::base::get<std::shared_ptr<int>>(smart_ptr_any);
  std::cout << "智能指针值: " << *retrieved_ptr << std::endl;

  // 测试传入一个指针
  int* ptr2 = new int(43);
  nndeploy::base::Any ptr_any;
  ptr_any.construct<int*>(ptr2);
  int* retrieved_ptr2 = nndeploy::base::get<int*>(ptr_any);
  std::cout << "指针值: " << *retrieved_ptr2 << std::endl;
  delete retrieved_ptr2;

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