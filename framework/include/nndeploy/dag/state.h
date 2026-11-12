#ifndef _NNDEPLOY_DAG_STATE_H_
#define _NNDEPLOY_DAG_STATE_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "nndeploy/base/any.h"
#include "nndeploy/base/status.h"

namespace nndeploy {
namespace dag {

// Request-owned state. The lock protects slots, not the objects stored in them.
class NNDEPLOY_CC_API State {
 public:
  base::Status set(const std::string& key, const base::Any& value);
  base::Status get(const std::string& key, base::Any& value) const;
  bool contains(const std::string& key) const;
  bool erase(const std::string& key);

 private:
  mutable std::mutex mutex_;
  // Copy and destroy user values outside the lock (including Python objects).
  std::map<std::string, std::shared_ptr<const base::Any>> values_;
};

}  // namespace dag
}  // namespace nndeploy
#endif
