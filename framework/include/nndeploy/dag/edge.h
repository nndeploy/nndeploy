#ifndef _NNDEPLOY_DAG_EDGE_H_
#define _NNDEPLOY_DAG_EDGE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/base.h"
#include "nndeploy/dag/edge/abstract_edge.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

/**
 * @brief Edge class in DAG graph for connecting nodes and transferring data
 *
 * The Edge class is one of the core components of the DAG graph in the nndeploy
 * framework, responsible for transferring data between nodes. It provides
 * functionality similar to std::any, capable of storing arbitrary types of data
 * while supporting advanced features such as memory management, queue
 * buffering, and parallel processing.
 *
 * It is recommended to use template functions
 * set/create/notifyWritten/get/getGraphOutput to set and retrieve data,
 */
class NNDEPLOY_CC_API Edge : public base::NonCopyable {
 public:
  /**
   * @brief Default constructor
   *
   * Creates an unnamed Edge instance with default configuration parameters
   */
  Edge();

  /**
   * @brief Constructor with name
   *
   * @param name Name of the Edge, used for debugging and logging
   */
  Edge(const std::string &name);

  /**
   * @brief Virtual destructor
   *
   * Automatically cleans up internal resources, including abstract edge
   * instances and related data
   */
  virtual ~Edge();

  /**
   * @brief Get the name of the Edge
   *
   * @return std::string Name string of the Edge
   */
  std::string getName();

  /**
   * @brief Set the maximum queue capacity
   *
   * Controls the maximum capacity of the edge buffer, used for data buffering
   * in pipeline parallel processing. When the queue is full, the overflow
   * policy determines how to handle new data.
   *
   * @param queue_max_size Maximum capacity of the queue, must be greater than
   * 0, default is 16
   * @return base::Status Returns kSuccess on success, kInvalidParam for invalid
   * parameters
   * @note Must be set before calling construct(), otherwise returns
   * kUninitialized
   */
  base::Status setQueueMaxSize(int queue_max_size);

  /**
   * @brief Get the maximum queue capacity
   *
   * @return int Current maximum capacity of the queue, default value is 16
   */
  int getQueueMaxSize();

  /**
   * @brief Set the queue overflow policy
   *
   * Processing strategy when the queue is full, supports different strategies
   * like backpressure, dropping, etc.
   *
   * @param policy Overflow handling policy
   * @param drop_count Number of data items to drop each time when policy is
   * drop, default is 1
   * @return base::Status Operation status
   */
  base::Status setQueueOverflowPolicy(base::QueueOverflowPolicy policy,
                                      int drop_count = 1);

  /**
   * @brief Get the current queue overflow policy
   *
   * @return base::QueueOverflowPolicy Current overflow handling policy
   */
  base::QueueOverflowPolicy getQueueOverflowPolicy();

  /**
   * @brief Get the queue drop count
   *
   * @return int Number of data items dropped each time on overflow
   */
  int getQueueDropCount();

  /**
   * @brief Set the parallel type
   *
   * Sets the parallel processing type of the Edge, affecting the creation and
   * management of internal data structures
   *
   * @param paralle_type Parallel type (serial, pipeline parallel, etc.)
   * @return base::Status Operation status
   * @note Must be called before construct(), internally creates corresponding
   * concrete edge implementation
   */
  base::Status setParallelType(const base::ParallelType &paralle_type);

  /**
   * @brief Get the current parallel type
   *
   * @return base::ParallelType Current parallel processing type
   */
  base::ParallelType getParallelType();

  /**
   * @brief Check if the Edge is empty
   *
   * @return bool true indicates no data in the Edge, false indicates data
   * exists
   */
  bool empty();

  /**
   * @brief Construct the Edge instance
   *
   * Creates concrete AbstractEdge implementation based on set parallel type and
   * other parameters
   *
   * @return base::Status Construction status, returns kSuccess on success
   */
  base::Status construct();

  // ==================== Buffer Related Interfaces ====================

  /**
   * @brief Set Buffer data to Edge
   *
   * @param buffer Buffer pointer
   * @param is_external Whether it's external data, true means not responsible
   * for memory deallocation
   * @return base::Status Operation status
   */
  base::Status set(device::Buffer *buffer, bool is_external = true);

  /**
   * @brief Set Buffer reference to Edge
   *
   * @param buffer Buffer reference, automatically set as external data
   * @return base::Status Operation status
   */
  base::Status set(device::Buffer &buffer);

  /**
   * @brief Create Buffer on specified device
   *
   * @param device Target device
   * @param desc Buffer description information
   * @return device::Buffer* Created Buffer pointer, returns nullptr on failure
   */
  device::Buffer *create(device::Device *device,
                         const device::BufferDesc &desc);

  /**
   * @brief Notify that Buffer data has been written
   *
   * @param buffer Written Buffer pointer
   * @return bool Whether notification was successful
   */
  bool notifyWritten(device::Buffer *buffer);

  /**
   * @brief Get Buffer data for specified node
   *
   * @param node Node requesting data
   * @return device::Buffer* Buffer pointer, returns nullptr if no data
   */
  device::Buffer *getBuffer(const Node *node);

  /**
   * @brief Get Buffer data for graph output
   *
   * @return device::Buffer* Buffer pointer, returns nullptr if no data
   */
  device::Buffer *getGraphOutputBuffer();

#ifdef ENABLE_NNDEPLOY_OPENCV
  // ==================== OpenCV Mat Related Interfaces ====================

  /**
   * @brief Set OpenCV Mat data to Edge
   *
   * @param cv_mat OpenCV Mat pointer
   * @param is_external Whether it's external data
   * @return base::Status Operation status
   */
  base::Status set(cv::Mat *cv_mat, bool is_external = true);

  /**
   * @brief Set OpenCV Mat reference to Edge
   *
   * @param cv_mat OpenCV Mat reference
   * @return base::Status Operation status
   */
  base::Status set(cv::Mat &cv_mat);

  /**
   * @brief Create OpenCV Mat with specified size and type
   *
   * @param rows Number of rows
   * @param cols Number of columns
   * @param type OpenCV data type
   * @param value Initialization value
   * @return cv::Mat* Created Mat pointer, returns nullptr on failure
   */
  cv::Mat *create(int rows, int cols, int type, const cv::Vec3b &value);

  /**
   * @brief Notify that OpenCV Mat data has been written
   *
   * @param cv_mat Written Mat pointer
   * @return bool Whether notification was successful
   */
  bool notifyWritten(cv::Mat *cv_mat);

  /**
   * @brief Get OpenCV Mat data for specified node
   *
   * @param node Node requesting data
   * @return cv::Mat* Mat pointer, returns nullptr if no data
   */
  cv::Mat *getCvMat(const Node *node);

  /**
   * @brief Get OpenCV Mat data for graph output
   *
   * @return cv::Mat* Mat pointer, returns nullptr if no data
   */
  cv::Mat *getGraphOutputCvMat();
#endif

  // ==================== Tensor Related Interfaces ====================

  /**
   * @brief Set Tensor data to Edge
   *
   * @param tensor Tensor pointer
   * @param is_external Whether it's external data
   * @return base::Status Operation status
   */
  base::Status set(device::Tensor *tensor, bool is_external = true);

  /**
   * @brief Set Tensor reference to Edge
   *
   * @param tensor Tensor reference
   * @return base::Status Operation status
   */
  base::Status set(device::Tensor &tensor);

  /**
   * @brief Create Tensor on specified device
   *
   * @param device Target device
   * @param desc Tensor description information
   * @param tensor_name Tensor name, optional
   * @return device::Tensor* Created Tensor pointer, returns nullptr on failure
   */
  device::Tensor *create(device::Device *device, const device::TensorDesc &desc,
                         std::string tensor_name = "");

  /**
   * @brief Notify that Tensor data has been written
   *
   * @param tensor Written Tensor pointer
   * @return bool Whether notification was successful
   */
  bool notifyWritten(device::Tensor *tensor);

  /**
   * @brief Get Tensor data for specified node
   *
   * @param node Node requesting data
   * @return device::Tensor* Tensor pointer, returns nullptr if no data
   */
  device::Tensor *getTensor(const Node *node);

  /**
   * @brief Get Tensor data for graph output
   *
   * @return device::Tensor* Tensor pointer, returns nullptr if no data
   */
  device::Tensor *getGraphOutputTensor();

  // ==================== Param Related Interfaces ====================

  /**
   * @brief Set Param data to Edge
   *
   * @param param Param pointer
   * @param is_external Whether it's external data
   * @return base::Status Operation status
   */
  base::Status set(base::Param *param, bool is_external = true);

  /**
   * @brief Set Param reference to Edge
   *
   * @param param Param reference
   * @return base::Status Operation status
   */
  base::Status set(base::Param &param);

  /**
   * @brief Create Param object of specified type
   *
   * @tparam T Concrete type of Param, must inherit from base::Param
   * @tparam Args Constructor parameter types
   * @param args Constructor parameters
   * @return base::Param* Created Param pointer, returns nullptr on failure
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<base::Param, T>::value,
                                    int>::type = 0>
  base::Param *create(Args &&...args) {
    this->setTypeInfo<T>();
    return abstact_edge_->create<T>(std::forward<Args>(args)...);
  }

  /**
   * @brief Notify that Param data has been written
   *
   * @param param Written Param pointer
   * @return bool Whether notification was successful
   */
  bool notifyWritten(base::Param *param);

  /**
   * @brief Get Param data for specified node
   *
   * @param node Node requesting data
   * @return base::Param* Param pointer, returns nullptr if no data
   */
  base::Param *getParam(const Node *node);

  /**
   * @brief Get Param data for graph output
   *
   * @return base::Param* Param pointer, returns nullptr if no data
   */
  base::Param *getGraphOutputParam();

  /**
   * @brief Get raw pointer for graph output
   *
   * @return void* Raw data pointer, type conversion is user's responsibility
   */
  void *getGraphOutputPtr();

  // ==================== Generic Template Interfaces ====================

  /**
   * @brief Set arbitrary type data to Edge (template version)
   *
   * @tparam T Data type
   * @param t Data pointer
   * @param is_external Whether it's external data
   * @return base::Status Operation status
   */
  template <typename T>
  base::Status set(T *t, bool is_external = true) {
    this->setTypeInfo<T>();
    return abstact_edge_->set<T>(t, is_external);
  }

  /**
   * @brief Set arbitrary type data reference to Edge (template version)
   *
   * @tparam T Data type
   * @param t Data reference
   * @return base::Status Operation status
   */
  template <typename T>
  base::Status set(T &t) {
    this->setTypeInfo<T>();
    return this->set(&t, true);
  }

  /**
   * @brief Create arbitrary type data object (template version)
   *
   * @tparam T Data type
   * @tparam Args Constructor parameter types
   * @param args Constructor parameters
   * @return T* Created object pointer, returns nullptr on failure
   */
  template <typename T, typename... Args>
  T *create(Args &&...args) {
    this->setTypeInfo<T>();
    return abstact_edge_->create<T>(std::forward<Args>(args)...);
  }

  /**
   * @brief Notify that arbitrary type data has been written (template version)
   *
   * @tparam T Data type
   * @param t Written data pointer
   * @return bool Whether notification was successful
   */
  template <typename T>
  bool notifyWritten(T *t) {
    return abstact_edge_->notifyWritten<T>(t);
  }

  /**
   * @brief Get arbitrary type data for specified node (template version)
   *
   * In pipeline parallel mode, waits for type information to be available
   * before returning data
   *
   * @tparam T Data type
   * @param node Node requesting data
   * @return T* Data pointer, returns nullptr if type mismatch or no data
   */
  template <typename T>
  T *get(const Node *node) {
    if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
      std::unique_lock<std::mutex> lock(type_info_mutex_);
      type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
    }
    if (type_info_ != nullptr && !type_info_->isType<T>()) {
      // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
      return nullptr;
    }
    return abstact_edge_->get<T>(node);
  }

  /**
   * @brief Get arbitrary type data for graph output (template version)
   *
   * In pipeline parallel mode, waits for type information to be available
   * before returning data
   *
   * @tparam T Data type
   * @return T* Data pointer, returns nullptr if type mismatch or no data
   */
  template <typename T>
  T *getGraphOutput() {
    if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
      std::unique_lock<std::mutex> lock(type_info_mutex_);
      type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
    }
    if (type_info_ != nullptr && !type_info_->isType<T>()) {
      // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
      return nullptr;
    }
    return abstact_edge_->getGraphOutput<T>();
  }

  /**
   * @brief Data setting interface dedicated for Python binding
   *
   * @tparam PY_WRAPPER Python wrapper type
   * @tparam T Actual data type
   * @param wrapper Python wrapper pointer
   * @param t Data pointer
   * @param is_external Whether it's external data
   * @return base::Status Operation status
   */
  template <typename PY_WRAPPER, typename T>
  base::Status set4py(PY_WRAPPER *wrapper, T *t, bool is_external = true) {
    this->setTypeInfo<T>();
    return abstact_edge_->set4py<PY_WRAPPER, T>(wrapper, t, is_external);
  }

  // ==================== Index and Position Management ====================

  /**
   * @brief Get data index for specified node
   *
   * @param node Target node
   * @return int64_t Data index value
   */
  int64_t getIndex(const Node *node);

  /**
   * @brief Get data index for graph output
   *
   * @return int64_t Data index value
   */
  int64_t getGraphOutputIndex();

  /**
   * @brief Reset index counter
   */
  void resetIndex();

  /**
   * @brief Get position of specified node in queue
   *
   * @param node Target node
   * @return int Queue position
   */
  int getPosition(const Node *node);

  /**
   * @brief Get position of graph output in queue
   *
   * @return int Queue position
   */
  int getGraphOutputPosition();

  /**
   * @brief Update Edge status for specified node
   *
   * @param node Target node
   * @return base::EdgeUpdateFlag Update flag
   */
  base::EdgeUpdateFlag update(const Node *node);

  /**
   * @brief Mark as graph output Edge
   *
   * @return bool Whether marking was successful
   * @note Must be called after graph initialization is complete
   */
  bool markGraphOutput();

  // ==================== Producer-Consumer Management ====================

  /**
   * @brief Add producer nodes
   *
   * @param producers List of producer nodes
   * @return base::Status Operation status
   */
  base::Status increaseProducers(std::vector<Node *> &producers);

  /**
   * @brief Add consumer nodes
   *
   * @param consumers List of consumer nodes
   * @return base::Status Operation status
   */
  base::Status increaseConsumers(std::vector<Node *> &consumers);

  /**
   * @brief Get all producer nodes
   *
   * @return std::vector<Node *> List of producer nodes
   */
  std::vector<Node *> getProducers();

  /**
   * @brief Get all consumer nodes
   *
   * @return std::vector<Node *> List of consumer nodes
   */
  std::vector<Node *> getConsumers();

  /**
   * @brief Request termination of Edge processing
   *
   * @return bool Whether request was successful
   */
  bool requestTerminate();

  // ==================== Type Information Management ====================

  /**
   * @brief Set type information of Edge (template version)
   *
   * @tparam T Data type
   * @return base::Status Operation status
   */
  template <typename T>
  base::Status setTypeInfo() {
    if (type_info_ == nullptr) {
      type_info_ = std::make_shared<EdgeTypeInfo>();
      type_info_->setType<T>();
    } else {
      type_info_->setType<T>();
    }
    if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
      type_info_cv_.notify_all();
    }
    return base::kStatusCodeOk;
  }

  /**
   * @brief Set type information of Edge
   *
   * @param type_info Type information object
   * @return base::Status Operation status
   */
  base::Status setTypeInfo(std::shared_ptr<EdgeTypeInfo> type_info);

  /**
   * @brief Get type information of Edge
   *
   * @return std::shared_ptr<EdgeTypeInfo> Type information object
   */
  std::shared_ptr<EdgeTypeInfo> getTypeInfo();

  /**
   * @brief Set type name
   *
   * @param type_name Type name string
   */
  void setTypeName(const std::string &type_name);

  /**
   * @brief Get type name
   *
   * @return std::string Type name string
   */
  std::string getTypeName();

  /**
   * @brief Check if type information matches (template version)
   *
   * @tparam T Data type to check
   * @return bool true indicates type matches, false indicates mismatch
   */
  template <typename T>
  bool checkTypeInfo() {
    EdgeTypeInfo other_type_info;
    other_type_info.setType<T>();
    return *type_info_ == other_type_info;
  }

  /**
   * @brief Check if type information matches
   *
   * @param type_info Type information to check
   * @return bool true indicates type matches, false indicates mismatch
   */
  bool checkTypeInfo(std::shared_ptr<EdgeTypeInfo> type_info);

 private:
  std::string name_;  ///< Edge name
  AbstractEdge *abstact_edge_ =
      nullptr;                  ///< Abstract edge implementation pointer
  std::mutex type_info_mutex_;  ///< Type information mutex
  std::condition_variable
      type_info_cv_;  ///< Type information condition variable
  std::shared_ptr<EdgeTypeInfo> type_info_;  ///< Type information object
  int queue_max_size_ = 16;  ///< Maximum queue capacity, default 16
  /// Queue overflow policy, default is node backpressure
  base::QueueOverflowPolicy queue_overflow_policy_ =
      base::QueueOverflowPolicy::kQueueOverflowPolicyNodeBackpressure;
  int queue_drop_count_ =
      1;  ///< Number of data items to drop on queue overflow, default 1
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EDGE_V2_H_ */
