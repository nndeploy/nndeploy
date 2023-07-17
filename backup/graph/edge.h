#ifndef _NNDEPLOY_INCLUDE_GRAPH_EDGE_H_
#define _NNDEPLOY_INCLUDE_GRAPH_EDGE_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/glic_stl_include.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/param.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/buffer_pool.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/device/mat.h"
#include "nndeploy/include/device/tensor.h"

namespace nndeploy {
namespace graph {

class NNDEPLOY_CC_API Edge {
 public:
  Edge();
  Edge(std::string name);

  Edge(const std::vector<device::Buffer *> &buffers);
  Edge(device::Buffer *buffer);
  Edge(device::Buffer &buffer);

  Edge(const std::vector<device::Mat *> &mats);
  Edge(device::Mat *mat);
  Edge(device::Mat &mat);

  Edge(const std::vector<device::Tensor *> &tensors);
  Edge(device::Tensor *tensor);
  Edge(device::Tensor &tensor);

  Edge(const std::vector<base::Param *> &param);
  Edge(base::Param *param);
  Edge(base::Param &param);

#ifdef NNDEPLOY_ENABLE_OPENCV
  Edge(const std::vector<cv::Mat *> &cv_mats);
  Edge(cv::Mat *cv_mat);
  Edge(cv::Mat &cv_mat);
#endif

  virtual ~Edge();

  base::Status setName(std::string name);
  std::string getName();

  void add(const std::vector<device::Buffer *> &buffers);
  void add(device::Buffer *buffer);
  void add(device::Buffer &buffer);
  void remove(const std::vector<device::Buffer *> &buffers);
  void remove(device::Buffer *buffer);
  void remove(device::Buffer &buffer);
  void clearBuffer();

  void add(const std::vector<device::Mat *> &mats);
  void add(device::Mat *mat);
  void add(device::Mat &mat);
  void remove(const std::vector<device::Mat *> &mats);
  void remove(device::Mat *mat);
  void remove(device::Mat &mat);
  void clearMat();

  void add(const std::vector<device::Tensor *> &tensors);
  void add(device::Tensor *tensor);
  void add(device::Tensor &tensor);
  void remove(const std::vector<device::Tensor *> &tensors);
  void remove(device::Tensor *tensor);
  void remove(device::Tensor &tensor);
  void clearTensor();

  void add(const std::vector<base::Param *> &param);
  void add(base::Param *param);
  void add(base::Param &param);
  void remove(const std::vector<base::Param *> &param);
  void remove(base::Param *param);
  void remove(base::Param &param);
  void clearParam();

#ifdef NNDEPLOY_ENABLE_OPENCV
  void add(const std::vector<cv::Mat *> &cv_mats);
  void add(cv::Mat *cv_mat);
  void add(cv::Mat &cv_mat);
  void remove(const std::vector<cv::Mat *> &cv_mats);
  void remove(cv::Mat *cv_mat);
  void remove(cv::Mat &cv_mat);
  void clearCvMat();
#endif

  void add(const std::vector<void *> &anythings);
  void add(void *anything);
  void remove(const std::vector<void *> &anythings);
  void remove(void *anything);
  void clearAnything();

  void clear();

  bool empty();

  bool emptyBuffer();
  int getBufferSize();
  device::Buffer *getBuffer();
  device::Buffer *getBuffer(int index);

  bool emptyMat();
  int getMatSize();
  device::Mat *getMat();
  device::Mat *getMat(int index);

  bool emptyTensor();
  int getTensorSize();
  device::Tensor *getTensor();
  device::Tensor *getTensor(int index);

  bool emptyParam();
  int getParamSize();
  base::Param *getParam();
  base::Param *getParam(int index);

#ifdef NNDEPLOY_ENABLE_OPENCV
  bool emptyCvMat();
  int getCvMatSize();
  cv::Mat *getCvMat();
  cv::Mat *getCvMat(int index);
#endif

  void *getAnything();

 private:
  std::string name_;

  std::vector<device::Buffer *> buffers_;
  std::vector<device::Mat *> mats_;
  std::vector<device::Tensor *> tensors_;

  std::vector<base::Param *> params_;

#ifdef NNDEPLOY_ENABLE_OPENCV
  std::vector<cv::Mat *> cv_mats_;
#endif

  std::vector<void *> anythings_;
};

}  // namespace graph
}  // namespace nndeploy

#endif /* _NNDEPLOY_INCLUDE_GRAPH_EDGE_H_ */
