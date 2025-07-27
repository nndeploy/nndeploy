#ifndef _NNDEPLOY_SEGMENT_SAM_H_
#define _NNDEPLOY_SEGMENT_SAM_H_
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
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/params.h"

namespace nndeploy {
namespace segment {

class NNDEPLOY_CC_API SAMPointsParam : public base::Param {
 public:
  std::vector<float> points_;
  std::vector<float> labels_;
  int ori_width;
  int ori_height;
  int version_ = -1;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json);
};

class SelectPointNode : public dag::Node {
 public:
  SelectPointNode(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::segment::SelectPointNode";
    desc_ = "Segment Anything Select Point Node for image segmentation tasks.";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<segment::SAMPointsParam>();
    param_ = std::make_shared<segment::SAMPointsParam>();
  }
  SelectPointNode(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::SelectPointNode";
    desc_ = "Segment Anything Select Point Node for image segmentation tasks.";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<segment::SAMPointsParam>();
    param_ = std::make_shared<segment::SAMPointsParam>();
  }
  virtual ~SelectPointNode() {}

  base::Status setPoints(const std::vector<float> &points,
                         const std::vector<float> &point_lables) {
    if (points.size() % 2 != 0) {
      NNDEPLOY_LOGE("Points should be in pairs (x, y).");
      return base::kStatusCodeErrorInvalidValue;
    }
    if (points.size() / 2 != point_lables.size()) {
      NNDEPLOY_LOGE("Number of points and point labels must match.");
      return base::kStatusCodeErrorInvalidValue;
    }

    points_ = points;
    point_labels_ = point_lables;

    return base::kStatusCodeOk;
  }

  base::Status run() override {
    // This node is a placeholder for selecting points.
    // In a real implementation, you would interact with the user to select
    // points. Here we just log the input image size.
    if (this->inputs_.empty()) {
      NNDEPLOY_LOGE("No input image provided.");
      return base::kStatusCodeErrorInvalidValue;
    }
    cv::Mat *input_image = inputs_[0]->getCvMat(this);

    segment::SAMPointsParam *sam_points_param = new segment::SAMPointsParam();
    sam_points_param->ori_height = input_image->rows;
    sam_points_param->ori_width = input_image->cols;

    if (points_.empty() == false && point_labels_.empty() == false) {
      sam_points_param->points_ = points_;
      sam_points_param->labels_ = point_labels_;
    } else {
      segment::SAMPointsParam *param =
          dynamic_cast<segment::SAMPointsParam *>(param_.get());
      sam_points_param->points_ = param->points_;
      sam_points_param->labels_ = param->labels_;
    }

    outputs_[0]->set(sam_points_param, false);

    return base::kStatusCodeOk;
  }

 private:
  std::vector<float> points_;
  std::vector<float> point_labels_;
};

/**
 * 当前集成版本默认无mask
 */
class NNDEPLOY_CC_API SAMGraph : public dag::Graph {
 public:
  SAMGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::segment::SAMGraph";
    desc_ = "Segment Anything Graph for image segmentation tasks.";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<SAMPointsParam>();
    this->setOutputTypeInfo<cv::Mat>();
    initGraphNodes();
  }

  SAMGraph(const std::string &name, std::vector<dag::Edge *> inputs,
           std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::segment::SAMGraph";
    desc_ = "Segment Anything Graph for image segmentation tasks.";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<SAMPointsParam>();
    this->setOutputTypeInfo<cv::Mat>();
    initGraphNodes();
  }

  // base::Status init() override;

  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::vector<std::string> &model_value);

 private:
  base::Status initGraphNodes();

 private:
  dag::Node *preprocess_image_node_ = nullptr;
  dag::Node *preprocess_point_node_ = nullptr;
  dag::Node *preprocess_mask_node_ = nullptr;

  infer::Infer *encoder_infer_node_ = nullptr;
  inference::InferenceParam encoder_infer_param_;
  infer::Infer *decoder_infer_node_ = nullptr;
  inference::InferenceParam decoder_infer_param_;

  dag::Node *postprocess_node_ = nullptr;
};

}  // namespace segment
}  // namespace nndeploy

#endif