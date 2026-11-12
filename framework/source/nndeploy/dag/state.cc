#include "nndeploy/dag/state.h"

namespace nndeploy {
namespace dag {

base::Status State::set(const std::string& key, const base::Any& value) {
  auto replacement = std::make_shared<const base::Any>(value);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    values_[key].swap(replacement);
  }
  return base::kStatusCodeOk;
}

base::Status State::get(const std::string& key, base::Any& value) const {
  std::shared_ptr<const base::Any> stored;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = values_.find(key);
    if (it == values_.end()) return base::kStatusCodeErrorInvalidValue;
    stored = it->second;
  }
  value = *stored;
  return base::kStatusCodeOk;
}

bool State::contains(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return values_.count(key) != 0;
}

bool State::erase(const std::string& key) {
  std::shared_ptr<const base::Any> removed;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = values_.find(key);
    if (it == values_.end()) return false;
    removed = std::move(it->second);
    values_.erase(it);
  }
  return true;
}

}  // namespace dag
}  // namespace nndeploy
