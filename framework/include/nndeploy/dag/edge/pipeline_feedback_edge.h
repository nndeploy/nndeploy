// #ifndef _NNDEPLOY_DAG_EDGE_PIPELINE_FEEDBACK_EDGE_H_
// #define _NNDEPLOY_DAG_EDGE_PIPELINE_FEEDBACK_EDGE_H_

// #include <atomic>
// #include <condition_variable>
// #include <mutex>
// #include <unordered_map>
// #include <vector>

// #include "nndeploy/base/common.h"
// #include "nndeploy/base/glic_stl_include.h"
// #include "nndeploy/base/log.h"
// #include "nndeploy/base/macro.h"
// #include "nndeploy/base/object.h"
// #include "nndeploy/base/opencv_include.h"
// #include "nndeploy/base/param.h"
// #include "nndeploy/base/spmc_ring.h"
// #include "nndeploy/base/status.h"
// #include "nndeploy/dag/edge/abstract_edge.h"
// #include "nndeploy/dag/node.h"
// #include "nndeploy/device/buffer.h"
// #include "nndeploy/device/device.h"
// #include "nndeploy/device/memory_pool.h"
// #include "nndeploy/device/tensor.h"

// namespace nndeploy {
// namespace dag {

// class PipelineFeedbackEdge : public AbstractEdge {
//  public:
//   PipelineFeedbackEdge(base::ParallelType parallel_type);
//   ~PipelineFeedbackEdge() override;

//   // lifetime
//   base::Status construct() override;
//   base::Status setQueueMaxSize(int q) override;

//   base::Status set(device::Buffer* buf, bool external) override;
//   device::Buffer* create(device::Device* dev,
//                          const device::BufferDesc& desc) override;
//   bool notifyWritten(device::Buffer* buf) override;

//   base::Status set(device::Tensor* ten, bool external) override;
//   device::Tensor* create(device::Device* dev, const device::TensorDesc& desc,
//                          const std::string& tensor_name) override;
//   bool notifyWritten(device::Tensor* ten) override;

//   base::Status set(base::Param* p, bool external) override;
//   bool notifyWritten(base::Param* p) override;

// #ifdef ENABLE_NNDEPLOY_OPENCV
//   base::Status set(cv::Mat* m, bool external) override;
//   cv::Mat* create(int rows, int cols, int type,
//                   const cv::Vec3b& value) override;
//   bool notifyWritten(cv::Mat* m) override;
// #endif

//   device::Buffer* getBuffer(const Node* n) override;
//   device::Buffer* getGraphOutputBuffer() override;

//   device::Tensor* getTensor(const Node* n) override;
//   device::Tensor* getGraphOutputTensor() override;

//   base::Param* getParam(const Node* n) override;
//   base::Param* getGraphOutputParam() override;

// #ifdef ENABLE_NNDEPLOY_OPENCV
//   cv::Mat* getCvMat(const Node* n) override;
//   cv::Mat* getGraphOutputCvMat() override;
// #endif

//   // schedule
//   base::EdgeUpdateFlag update(const Node* n) override;
//   bool requestTerminate() override;
//   bool markGraphOutput() override;
//   bool hasBeenConsumedBy(const Node* n) override { return false; };

//   // debug
//   int64_t getIndex(const Node* n) override;
//   int64_t getGraphOutputIndex() override;
//   int getPosition(const Node* n) override;
//   int getGraphOutputPosition() override;

//  private:
//   using Ring = nndeploy::base::SpmcRing<void*, const void*>;

//   // SPMC Ring buffer
//   Ring ring_;
//   int queue_max_size_{1};

//   struct ConsumerState {
//     DataPacket* cur{nullptr};  // 当前 peek 到但未清理的包视图
//     bool committed{false};     // 是否已对 ring 执行过一次 pop/commit
//     int cid{SIZE_MAX};         // 在 ring 中的 reader id
//   };
//   std::unordered_map<const Node*, ConsumerState> states_;
//   std::vector<DataPacket*> pending_packets_;
//   int consumers_size_{0};
//   std::atomic<bool> terminated_{false};
// };

// }  // namespace dag
// }  // namespace nndeploy

// #endif