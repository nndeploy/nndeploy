// #include "nndeploy/dag/edge/pipeline_feedback_edge.h"

// #include "nndeploy/dag/edge/data_packet.h"

// namespace nndeploy {
// namespace dag {

// TypeEdgeRegister<TypeEdgeCreator<PipelineFeedbackEdge>>
//     g_pipeline_edge_register(base::kEdgeTypePipelineFeedback);

// PipelineFeedbackEdge::PipelineFeedbackEdge(base::ParallelType pt)
//     : AbstractEdge(pt) {}

// static inline const void* key_from_node(const Node* n) {
//   return static_cast<const void*>(n);
// }

// base::Status PipelineFeedbackEdge::construct() {
//   // 计算消费者数量（包含 graph output 的 nullptr）
//   consumers_size_ = static_cast<int>(consumers_.size());

//   // 重新配置 ring 容量并注册所有消费者
//   ring_.set_capacity(
//       queue_max_size_ > 0 ? static_cast<uint64_t>(queue_max_size_) : 1);

//   states_.clear();
//   for (auto* c : consumers_) {
//     // ring 以“key”区分消费者，这里直接用 Node*（nullptr 代表图输出）
//     ring_.register_consumer(c, /*start_from_latest=*/true);

//     ConsumerState st{};
//     st.cur = nullptr;
//     st.committed = false;
//     st.cid = 0;  // 未使用，仅占位
//     states_.emplace(c, st);
//   }

//   return base::kStatusCodeOk;
// }

// base::Status PipelineFeedbackEdge::setQueueMaxSize(int q) {
//   if (q <= 0) q = 1;
//   queue_max_size_ = q;

//   // 更改容量会 reset ring，需要重注 readers
//   // 这里直接复用 construct()
//   return construct();
// }

// base::Status PipelineFeedbackEdge::set(device::Buffer* buf, bool external) {
//   auto* dp = new FeedbackDataPacket();
//   NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp,
//                                        "alloc FeedbackDataPacket failed.\n");
//   this->increaseIndex();
//   dp->setIndex(this->index_);

//   base::Status s = dp->set(buf, external);
//   NNDEPLOY_RETURN_ON_NEQ(s, base::kStatusCodeOk, "dp->set(buffer)
//   failed.\n");

//   if (!ring_.push_blocking(static_cast<void*>(dp))) {
//     // 终止状态下 push 可能失败，回收
//     delete dp;
//     return base::kStatusCodeErrorInvalidValue;
//   }
//   return base::kStatusCodeOk;
// }

// device::Buffer* PipelineFeedbackEdge::create(device::Device* dev,
//                                              const device::BufferDesc& desc)
//                                              {
//   // 仅创建，不入环，待 notifyWritten
//   auto* dp = new FeedbackDataPacket();
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "alloc FeedbackDataPacket
//   failed.\n");

//   this->increaseIndex();
//   dp->setIndex(this->index_);

//   device::Buffer* b = dp->create(dev, desc);
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(b, "dp->create(buffer) failed.\n");

//   pending_packets_.push_back(dp);
//   return b;
// }

// bool PipelineFeedbackEdge::notifyWritten(device::Buffer* buf) {
//   // 在 pending 中查找匹配包，置 written 并入环
//   for (auto it = pending_packets_.rbegin(); it != pending_packets_.rend();
//        ++it) {
//     FeedbackDataPacket* dp = *it;
//     if (dp->notifyWritten(buf)) {
//       if (!ring_.push_blocking(static_cast<void*>(dp))) {
//         // push 失败（终止），回收并移除
//         delete dp;
//         pending_packets_.erase(std::next(it).base());
//         return false;
//       }
//       pending_packets_.erase(std::next(it).base());
//       return true;
//     }
//   }
//   NNDEPLOY_LOGE("notifyWritten(buffer=%p) not found pending packet.\n", buf);
//   return false;
// }

// base::Status PipelineFeedbackEdge::set(device::Tensor* ten, bool external) {
//   auto* dp = new FeedbackDataPacket();
//   NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp,
//                                        "alloc FeedbackDataPacket failed.\n");

//   this->increaseIndex();
//   dp->setIndex(this->index_);

//   base::Status s = dp->set(ten, external);
//   NNDEPLOY_RETURN_ON_NEQ(s, base::kStatusCodeOk, "dp->set(tensor)
//   failed.\n");

//   if (!ring_.push_blocking(static_cast<void*>(dp))) {
//     delete dp;
//     return base::kStatusCodeErrorInvalidValue;
//   }
//   return base::kStatusCodeOk;
// }

// device::Tensor* PipelineFeedbackEdge::create(device::Device* dev,
//                                              const device::TensorDesc& desc,
//                                              const std::string& name) {
//   auto* dp = new FeedbackDataPacket();
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "alloc FeedbackDataPacket
//   failed.\n");

//   this->increaseIndex();
//   dp->setIndex(this->index_);

//   device::Tensor* t = dp->create(dev, desc, name);
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(t, "dp->create(tensor) failed.\n");

//   pending_packets_.push_back(dp);
//   return t;
// }

// bool PipelineFeedbackEdge::notifyWritten(device::Tensor* ten) {
//   for (auto it = pending_packets_.rbegin(); it != pending_packets_.rend();
//        ++it) {
//     FeedbackDataPacket* dp = *it;
//     if (dp->notifyWritten(ten)) {
//       if (!ring_.push_blocking(static_cast<void*>(dp))) {
//         delete dp;
//         pending_packets_.erase(std::next(it).base());
//         return false;
//       }
//       pending_packets_.erase(std::next(it).base());
//       return true;
//     }
//   }
//   NNDEPLOY_LOGE("notifyWritten(tensor=%p) not found pending packet.\n", ten);
//   return false;
// }

// base::Status PipelineFeedbackEdge::set(base::Param* p, bool external) {
//   auto* dp = new FeedbackDataPacket();
//   NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp,
//                                        "alloc FeedbackDataPacket failed.\n");

//   this->increaseIndex();
//   dp->setIndex(this->index_);

//   base::Status s = dp->set(p, external);
//   NNDEPLOY_RETURN_ON_NEQ(s, base::kStatusCodeOk, "dp->set(param) failed.\n");

//   if (!ring_.push_blocking(static_cast<void*>(dp))) {
//     delete dp;
//     return base::kStatusCodeErrorInvalidValue;
//   }
//   return base::kStatusCodeOk;
// }

// bool PipelineFeedbackEdge::notifyWritten(base::Param* p) {
//   for (auto it = pending_packets_.rbegin(); it != pending_packets_.rend();
//        ++it) {
//     FeedbackDataPacket* dp = *it;
//     if (dp->notifyWritten(p)) {
//       if (!ring_.push_blocking(static_cast<void*>(dp))) {
//         delete dp;
//         pending_packets_.erase(std::next(it).base());
//         return false;
//       }
//       pending_packets_.erase(std::next(it).base());
//       return true;
//     }
//   }
//   NNDEPLOY_LOGE("notifyWritten(param=%p) not found pending packet.\n", p);
//   return false;
// }

// #ifdef ENABLE_NNDEPLOY_OPENCV
// base::Status PipelineFeedbackEdge::set(cv::Mat* m, bool external) {
//   auto* dp = new FeedbackDataPacket(consumers_size_);
//   NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp,
//                                        "alloc FeedbackDataPacket failed.\n");

//   this->increaseIndex();
//   dp->setIndex(this->index_);

//   base::Status s = dp->set(m, external);
//   NNDEPLOY_RETURN_ON_NEQ(s, base::kStatusCodeOk, "dp->set(cv::Mat)
//   failed.\n");

//   if (!ring_.push_blocking(static_cast<void*>(dp))) {
//     delete dp;
//     return base::kStatusCodeErrorInvalidState;
//   }
//   return base::kStatusCodeOk;
// }

// cv::Mat* PipelineFeedbackEdge::create(int rows, int cols, int type,
//                                       const cv::Vec3b& value) {
//   auto* dp = new FeedbackDataPacket(consumers_size_);
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "alloc FeedbackDataPacket
//   failed.\n");

//   this->increaseIndex();
//   dp->setIndex(this->index_);

//   cv::Mat* m = dp->create(rows, cols, type, value);
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(m, "dp->create(cv::Mat) failed.\n");

//   pending_packets_.push_back(dp);
//   return m;
// }

// bool PipelineFeedbackEdge::notifyWritten(cv::Mat* m) {
//   for (auto it = pending_packets_.rbegin(); it != pending_packets_.rend();
//        ++it) {
//     FeedbackDataPacket* dp = *it;
//     if (dp->notifyWritten(m)) {
//       if (!ring_.push_blocking(static_cast<void*>(dp))) {
//         delete dp;
//         pending_packets_.erase(std::next(it).base());
//         return false;
//       }
//       pending_packets_.erase(std::next(it).base());
//       return true;
//     }
//   }
//   NNDEPLOY_LOGE("notifyWritten(cv::Mat=%p) not found pending packet.\n", m);
//   return false;
// }
// #endif

// }  // namespace dag
// }  // namespace nndeploy