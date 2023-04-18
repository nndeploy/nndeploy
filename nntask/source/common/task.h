#ifndef _NNTASK_SOURCE_COMMON_TASK_
#define _NNTASK_SOURCE_COMMON_TASK_

namespace nntask {
namespace common {

class Task {
 public:
  Task() = default;
  virtual ~Task() = default;

  virtual base::Status setParam(Param* param) {
    params_.push_back(param);
    return base::kStatusCodeOk;
  }
  virtual base::Status setParam(std::vector<Param*> param) {
    params_.insert(params_.end(), param.begin(), param.end());
    return base::kStatusCodeOk;
  }

  virtual base::Status setInput(Edge* edge) {
    inputs_.push_back(edge);
    return base::kStatusCodeOk;
  }
  virtual base::Status setInput(std::vector<Edge*> edge) {
    inputs_.insert(inputs_.end(), edge.begin(), edge.end());
    return base::kStatusCodeOk;
  }
  virtual base::Status setOutput(Edge* edge) {
    inputs_.push_back(edge);
    return base::kStatusCodeOk;
  }
  virtual base::Status setOutput(std::vector<Edge*> edge) {
    outputs_.insert(outputs_.end(), edge.begin(), edge.end());
    return base::kStatusCodeOk;
  }

  virtual base::Status setDependElement(Element* element) {
    depend_elements_.push_back(element);
    return base::kStatusCodeOk;
  }
  virtual base::Status setDependElement(std::vector<Element*> element) {
    depend_elements_.insert(depend_elements_.end(), element.begin(),
                            element.end());
    return base::kStatusCodeOk;
  }

  virtual base::Status setDependTask(Task* task) {
    depend_tasks_.push_back(task);
    return base::kStatusCodeOk;
  }
  virtual base::Status setDependTask(std::vector<Task*> task) {
    depend_tasks_.insert(depend_tasks_.end(), task.begin(), task.end());
    return base::kStatusCodeOk;
  }

  virtual base::Status setDependTask(std::vector<Task*> task) {
    depend_tasks_.insert(depend_tasks_.end(), task.begin(), task.end());
    return base::kStatusCodeOk;
  }

  virtual base::Status setDependTask(std::vector<Task*> task) {
    depend_tasks_.insert(depend_tasks_.end(), task.begin(), task.end());
    return base::kStatusCodeOk;
  }

  virtual base::Status setDependTask(std::vector<Task*> task) {
    depend_tasks_.insert(depend_tasks_.end(), task.begin(), task.end());
    return base::kStatusCodeOk;
  }
}
}  // namespace common
}  // namespace nntask

#endif /* _NNTASK_SOURCE_COMMON_TASK_ */
