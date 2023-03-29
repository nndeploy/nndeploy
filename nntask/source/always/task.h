
#ifndef _NNTASK_SOURCE_ALWAYS_TASK_H_
#define _NNTASK_SOURCE_ALWAYS_TASK_H_

#include "nndeploy/source/base/include_c_cpp.h"

namespace nntask {
namespace always {

class Task {
 public:
  Task() {}
  ~Task() {}

  int add(int a, int b) { return a + b; }

  int sub(int a, int b);
};

}  // namespace always
}  // namespace nntask

#endif
