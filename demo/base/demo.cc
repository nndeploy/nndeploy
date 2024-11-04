#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "nndeploy/base/any.h"

int main(int argc, char* argv[]) { 
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

  return 0;
}