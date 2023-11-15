
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
#include "nndeploy/dag/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

class Node;

enum EdgeFlag : int {
  kEdgeFlagBuffer,
  kEdgeFlagMat = 1,
#ifdef ENABLE_NNDEPLOY_OPENCV
  kEdgeFlagCvMat = 2,
#endif
  kEdgeFlagTensor = 4,
  kEdgeFlagParam = 8,

  kEdgeFlagVoid = 1 << 30,

  kEdgeFlagNone = 1 << 31,
};

class NNDEPLOY_CC_API AbstractEdge : public base::NonCopyable {
 public:
  AbstractEdge() {}
  virtual ~AbstractEdge() {}
  virtual base::Status set(device::Buffer *buffer, bool is_external = true,
                           int pts = -1) = 0;
  virtual base::Status set(device::Buffer &buffer, bool is_external = true,
                           int pts = -1) = 0;
  virtual base::Status set(device::Mat *mat, bool is_external = true,
                           int pts = -1) = 0;
  virtual base::Status set(device::Mat &mat, bool is_external = true,
                           int pts = -1) = 0;
#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, bool is_external = true,
                           int pts = -1) = 0;
  virtual base::Status set(cv::Mat &cv_mat, bool is_external = true,
                           int pts = -1) = 0;
#endif
  virtual base::Status set(device::Tensor *tensor, bool is_external = true,
                           int pts = -1) = 0;
  virtual base::Status set(device::Tensor &tensor, bool is_external = true,
                           int pts = -1) = 0;
  virtual base::Status set(base::Param *param, bool is_external = true,
                           int pts = -1) = 0;
  virtual base::Status set(base::Param &param, bool is_external = true,
                           int pts = -1) = 0;
  virtual base::Status set(void *anything, bool is_external = true,
                           int pts = -1) = 0;

  virtual base::Status create(device::Device *device,
                              const device::BufferDesc &desc) = 0;
  virtual base::Status create(device::Device *device,
                              const device::TensorDesc &desc,
                              const std::string &name) = 0;
  virtual base::Status create(device::Device *device,
                              const device::MatDesc &desc,
                              const std::string &name) = 0;

  virtual device::Buffer *getBuffer() = 0;
  virtual device::Mat *getMat() = 0;
#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual cv::Mat *getCvMat() = 0;
#endif
  virtual device::Tensor *getTensor() = 0;
  virtual base::Param *getParam() = 0;
  virtual base::Status *getAnything() = 0;

  bool isExternal() { return is_external_; }
  int getPts() { return pts_; }

 private:
};

class NNDEPLOY_CC_API FixedEdge : public AbstractEdge {
  void *anything_;
};

// typedef std::map<int, void *> ConsumedAnything;

class NNDEPLOY_CC_API PipelineEdge : public AbstractEdge {
  bool is_external_ = true;
  EdgeFlag flag_ = kEdgeFlagNone;
  int pts_ = -1;
  std::vector<Node *> producers_;
  std::vector<Node *> consumers_;

  thread_pool::SafeQueue<int> alread_consumed_;
  thread_pool::SafeQueue<void *> anything_;
  std::map<Node *, int> consumed_;
};

/**
 * @brief 需要保证Edge和Mat、Tensor名字一致
 *
 */
class NNDEPLOY_CC_API Edge : public base::NonCopyable {
 public:
  Edge() : name_(""), abstact_edge_(nullptr) {}
  Edge(const std::string &name) : name_(name), abstact_edge_(nullptr) {}
  virtual ~Edge() { delete abstact_edge_; }

  std::string getName() { return name_; }

  base::Status construct(ParallelType paralle_type,
                         std::initializer_list<Node *> producers,
                         std::initializer_list<Node *> consumers);

  base::Status set(device::Buffer *buffer, int index_ = -1,
                   bool is_external = true);
  base::Status set(device::Buffer &buffer, int index_ = -1,
                   bool is_external = true);
  base::Status create(device::Device *device, const device::BufferDesc &desc,
                      int index_ = -1);
  device::Buffer *getBuffer(const Node *comsumer);

  base::Status set(device::Mat *mat, int index_ = -1, bool is_external = true);
  base::Status set(device::Mat &mat, int index_ = -1, bool is_external = true);
  base::Status create(device::Device *device, const device::MatDesc &desc,
                      int index_ = -1);
  device::Mat *getMat(const Node *comsumer);

#ifdef ENABLE_NNDEPLOY_OPENCV
  base::Status set(cv::Mat *cv_mat, int index_ = -1, bool is_external = true);
  base::Status set(cv::Mat &cv_mat, int index_ = -1, bool is_external = true);
  cv::Mat *getCvMat(const Node *comsumer);
#endif

  base::Status set(device::Tensor *tensor, int index_ = -1,
                   bool is_external = true);
  base::Status set(device::Tensor &tensor, int index_ = -1,
                   bool is_external = true);
  base::Status create(device::Device *device, const device::TensorDesc &desc,
                      int index_ = -1);
  device::Tensor *getTensor(const Node *comsumer);

  base::Status set(base::Param *param, bool is_external = true, int pts = -1);
  base::Status set(base::Param &param, bool is_external = true, int pts = -1);
  base::Param *getParam(const Node *comsumer);

  base::Status set(void *anything, bool is_external = true, int pts = -1);
  void *getAnything(const Node *comsumer);

  int getIndex(const Node *comsumer);

 private:
  std::string name_;
  AbstractEdge *abstact_edge_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EDGE_V2_H_ */
