
#ifndef _NNTASK_SOURCE_DETR_TASK_H_
#define _NNTASK_SOURCE_DETR_TASK_H_

#include "nndeploy/source/base/glic_stl_include.h"
#include "nntask/source/common/task.h"

namespace nntask {
namespace detr {

class Task : public common::Task {
 public:
  Task(nndeploy::base::InferenceType type, std::string name);

  virtual nndeploy::base::Status setInput(common::Packet &input);
  virtual nndeploy::base::Status setOutput(common::Packet &output);
};

}  // namespace detr
}  // namespace nntask

#endif
