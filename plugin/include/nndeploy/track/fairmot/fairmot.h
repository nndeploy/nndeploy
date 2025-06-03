#ifndef _NNDEPLOY_TRACK_FAIRMOT_FAIRMOT_H_
#define _NNDEPLOY_TRACK_FAIRMOT_FAIRMOT_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/classification/result.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvtcolor_resize.h"
#include "nndeploy/preprocess/params.h"
#include "nndeploy/track/result.h"
#include "nndeploy/track/tracker.h"

namespace nndeploy {
namespace track {

struct TrailRecorder {
  std::map<int, std::vector<std::array<int, 2>>> records;
  void Add(int id, const std::array<int, 2>& record);
};

inline void TrailRecorder::Add(int id, const std::array<int, 2>& record) {
  auto iter = records.find(id);
  if (iter != records.end()) {
    auto trail = records[id];
    trail.push_back(record);
    records[id] = trail;
  } else {
    records[id] = {record};
  }
}

class NNDEPLOY_CC_API FairMotPostParam : public base::Param {
 public:
  float conf_thresh_ = 0.3f;
  float tracked_thresh_ = 0.5f;
  float min_box_area_ = 100.0f;
  bool is_record_trail_ = false;
};

class NNDEPLOY_CC_API FairMotPostProcess : public dag::Node {
 public:
  FairMotPostProcess(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::track::FairMotPostProcess";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<MOTResult>();
  }

  FairMotPostProcess(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::track::FairMotPostProcess";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<MOTResult>();
  }

  virtual ~FairMotPostProcess() {}

  virtual base::Status init();

  virtual base::Status deinit();

  virtual base::Status run();

  void FilterDets(const float conf_threshold, const cv::Mat& dets,
                  std::vector<int>* index);

 private:
  std::shared_ptr<JDETracker> jdeTracker_ = nullptr;
  std::shared_ptr<TrailRecorder> recorder_ = nullptr;
};

class NNDEPLOY_CC_API FairMotGraph : public dag::Graph {
 public:
  FairMotGraph(const std::string& name) : dag::Graph(name) {
    key_ = "nndeploy::track::FairMotGraph";
    this->setInputTypeInfo<cv::Mat>();
  }

  FairMotGraph(const std::string& name, std::vector<dag::Edge*> inputs,
               std::vector<dag::Edge*> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::track::FairMotGraph";
    this->setInputTypeInfo<cv::Mat>();
  }

  virtual ~FairMotGraph() {}

  base::Status make(const dag::NodeDesc& pre_desc,
                    const dag::NodeDesc& infer_desc,
                    base::InferenceType inference_type,
                    const dag::NodeDesc& post_desc) {
    pre_ = this->createNode<preprocess::CvtColorResize>(pre_desc);
    if (pre_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create preprocessing node\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    preprocess::CvtclorResizeParam* pre_param =
        dynamic_cast<preprocess::CvtclorResizeParam*>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 320;
    pre_param->w_ = 576;

    // Create inference node for fairmot model execution
    infer_ =
        dynamic_cast<infer::Infer*>(this->createNode<infer::Infer>(infer_desc));
    if (infer_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference node\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    infer_->setInferenceType(inference_type);

    // Create postprocessing node for tracking result
    post_ = this->createNode<FairMotPostProcess>(post_desc);
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    FairMotPostParam* post_param =
        dynamic_cast<FairMotPostParam*>(post_->getParam());

    return base::kStatusCodeOk;
  }

  base::Status setInferParam(base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::vector<std::string>& model_value) {
    auto param = dynamic_cast<inference::InferenceParam*>(infer_->getParam());
    param->device_type_ = device_type;
    param->model_type_ = model_type;
    param->is_path_ = is_path;
    param->model_value_ = model_value;
    return base::kStatusCodeOk;
  }

 private:
  dag::Node* pre_ = nullptr;
  infer::Infer* infer_ = nullptr;
  dag::Node* post_ = nullptr;
};

}  // namespace track
}  // namespace nndeploy

#endif