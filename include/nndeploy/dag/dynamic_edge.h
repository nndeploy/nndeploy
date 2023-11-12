#ifndef _NNDEPLOY_DAG_DYNAMIC_EDGE_H_
#define _NNDEPLOY_DAG_DYNAMIC_EDGE_H_
#include "nndeploy/dag/data_packet.h"
namespace nndeploy {
namespace dag {
class NNDEPLOY_CC_API DynamicEdge {
 public:
  DynamicEdge() {}
  ~DynamicEdge() {}

  base::Status setMaxSize(int max_size);
  int getMaxSize() { return max_size_; }

  base::Status enqueue(DataPacket *input);
  DataPacket *dequeue(int idx);

 private:
  int max_size_ = 1;
  std::vector<DataPacket *> data_ = {};
  std::mutex locker_;
  std::condition_variable cv_;
};
}  // namespace dag
}  // namespace nndeploy

#endif