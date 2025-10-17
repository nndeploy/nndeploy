
/**
 * nndeploy Base Framework Demo:
 * This example demonstrates the basic functionality of nndeploy framework,
 * focusing on the Any type system for type-safe generic value storage
 *
 * Main steps:
 * 1. Initialize nndeploy framework
 * 2. Test Any type with basic integer value construction and retrieval
 * 3. Test Any type with string value construction and retrieval
 * 4. Test Any type with smart pointer storage and retrieval
 * 5. Test Any type with raw pointer storage and retrieval
 * 6. Display framework version information
 * 7. Clean up and deinitialize framework
 */

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

  // 测试传入一个智能指针
  std::shared_ptr<int> ptr = std::make_shared<int>(42);
  nndeploy::base::Any smart_ptr_any;
  smart_ptr_any.construct<std::shared_ptr<int>>(ptr);
  std::shared_ptr<int> retrieved_ptr =
      nndeploy::base::get<std::shared_ptr<int>>(smart_ptr_any);
  std::cout << "智能指针值: " << *retrieved_ptr << std::endl;

  // 测试传入一个指针
  int* ptr2 = new int(43);
  nndeploy::base::Any ptr_any;
  ptr_any.construct<int*>(ptr2);
  int* retrieved_ptr2 = nndeploy::base::get<int*>(ptr_any);
  std::cout << "指针值: " << *retrieved_ptr2 << std::endl;
  delete retrieved_ptr2;

  return 0;
}