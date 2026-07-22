
#ifndef _NNDEPLOY_SEGMENT_SAM2_H_
#define _NNDEPLOY_SEGMENT_SAM2_H_

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

/**
 * @brief SAM 2 点提示参数类
 *
 * 与 SAM 1 的 SAMPointsParam 相同，但 version_ = 2
 * 支持点坐标 + 标签，用于交互式分割
 */
class NNDEPLOY_CC_API SAM2PointsParam : public base::Param {
 public:
  std::vector<float> points_;
  std::vector<float> labels_;
  int ori_width_;
  int ori_height_;
  int version_ = 2;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief SAM 2 后处理参数类
 *
 * 与 SAM 1 相同：从 decoder 输出的 masks + iou_predictions 中选择最佳 mask
 */
class NNDEPLOY_CC_API SAM2PostParam : public base::Param {
 public:
  float score_threshold_ = 0.0f;
  int model_h_ = 1024;
  int model_w_ = 1024;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief SAM 2 点预处理节点
 *
 * SAM 2 使用与 SAM 1 相同的点编码策略，
 * 将图像坐标映射到 1024x1024 模型空间。
 * 输出: point_coords, point_labels, orig_im_size
 */
class SAM2PointNode : public dag::Node {
 public:
  SAM2PointNode(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::SAM2PointNode";
    desc_ = "SAM 2 Point Node for point prompt encoding.";
    this->setInputTypeInfo<SAM2PointsParam>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    param_ = std::make_shared<SAM2PointsParam>();
    this->defaultParam();
  }
  SAM2PointNode(const std::string& name, std::vector<dag::Edge*> inputs,
                std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::SAM2PointNode";
    desc_ = "SAM 2 Point Node for point prompt encoding.";
    this->setInputTypeInfo<SAM2PointsParam>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    param_ = std::make_shared<SAM2PointsParam>();
    this->defaultParam();
  }
  virtual ~SAM2PointNode() {}

  base::Status defaultParam() override {
    SAM2PointsParam* param = dynamic_cast<SAM2PointsParam*>(param_.get());
    param->points_.clear();
    param->labels_.clear();
    param->ori_width_ = 0;
    param->ori_height_ = 0;
    param->version_ = 2;
    return base::kStatusCodeOk;
  }

  virtual base::Status run();
};

/**
 * @brief SAM 2 mask 预处理节点
 *
 * 生成空的 mask_input 和 has_mask_input tensor（默认无 mask）
 * 与 SAM 1 的 SAMMaskNode 功能相同
 */
class SAM2MaskNode : public dag::Node {
 public:
  SAM2MaskNode(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::SAM2MaskNode";
    desc_ = "SAM 2 Mask Node for mask input preparation.";
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  SAM2MaskNode(const std::string& name, std::vector<dag::Edge*> inputs,
               std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::SAM2MaskNode";
    desc_ = "SAM 2 Mask Node for mask input preparation.";
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  virtual ~SAM2MaskNode() {}

  virtual base::Status run();
};

/**
 * @brief SAM 2 后处理节点
 *
 * 与 SAM 1 的 SAMPostProcess 逻辑相同：
 * 从 decoder 的 masks 输出中选择 IoU 分数最高的 mask，
 * 转换为 CV_8UC1 输出。
 */
class SAM2PostProcess : public dag::Node {
 public:
  SAM2PostProcess(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::SAM2PostProcess";
    desc_ = "SAM 2 Post Process Node.";
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<cv::Mat>();
    param_ = std::make_shared<SAM2PostParam>();
    this->defaultParam();
  }
  SAM2PostProcess(const std::string& name, std::vector<dag::Edge*> inputs,
                  std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::SAM2PostProcess";
    desc_ = "SAM 2 Post Process Node.";
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<cv::Mat>();
    param_ = std::make_shared<SAM2PostParam>();
    this->defaultParam();
  }
  virtual ~SAM2PostProcess() {}

  virtual base::Status run();
  base::Status defaultParam() override;
};

/**
 * @brief SAM 2 视频记忆节点（掩码传播容器）
 *
 * 在视频序列帧之间存储和管理上一帧的输出掩码，
 * 用于掩码传播（mask propagation）实现时序一致的对象跟踪。
 *
 * 实现原理：
 * - Frame 0：解码器使用零掩码 + has_mask_input=0（无先验）
 * - Frame N+1：解码器使用上一帧输出掩码 + has_mask_input=1（时序传播）
 *
 * 注意：当前实现为简化的掩码传播（存储上一帧掩码数据）。
 * 完整的 SAM 2 视频跟踪需要独立的 memory_encoder ONNX 模型，
 * 该模型不在 vietanhdev ONNX 导出格式中。
 * 掩码传播提供基本的帧间一致性而不需要额外神经网络。
 */
class SAM2MemoryNode : public dag::Node {
 public:
  SAM2MemoryNode(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::SAM2MemoryNode";
    desc_ = "SAM 2 Memory Node for video tracking mask propagation.";
  }
  SAM2MemoryNode(const std::string& name, std::vector<dag::Edge*> inputs,
                 std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::SAM2MemoryNode";
    desc_ = "SAM 2 Memory Node for video tracking mask propagation.";
  }
  virtual ~SAM2MemoryNode() {}

  virtual base::Status run() { return base::kStatusCodeOk; }
  base::Status defaultParam() override { return base::kStatusCodeOk; }

  /// 存储解码器输出掩码（第一个通道，256x256）供下一帧使用
  void storeMask(device::Tensor* masks_tensor);

  /// 用存储的掩码填充边缘（mask_input + has_mask_input=1.0）
  void fillMaskEdge(dag::Edge* mask_edge, dag::Edge* has_mask_edge);

  /// 是否有上一帧的掩码可用于传播
  bool hasPrevMask() const {
    return frame_count_ > 0 && !prev_mask_data_.empty();
  }

  /// 清除记忆（新视频序列时调用）
  void reset() {
    frame_count_ = 0;
    prev_mask_data_.clear();
  }

 private:
  int frame_count_ = 0;
  std::vector<float> prev_mask_data_;
  int prev_mask_h_ = 256;
  int prev_mask_w_ = 256;
};

/**
 * @brief SAM 2 Graph - 图像/视频分割
 *
 * SAM 2 使用 Hiera backbone，encoder 输出多尺度特征：
 *   - image_embeddings: 主要特征 (1,256,64,64)
 *   - high_res_feats_0: 高分辨率特征 (1,32,256,256)
 *   - high_res_feats_1: 中分辨率特征 (1,64,128,128)
 *
 * Graph 输入：
 *   inputs[0]: 图像 (cv::Mat)
 *   inputs[1]: 点参数 (SAM2PointsParam)
 *
 * Graph 输出：
 *   outputs[0]: 分割 mask (cv::Mat)
 *
 * 架构：
 *   Image ──► preprocess_image ──► encoder_infer ──► image_embeddings ──┐
 *                                                      high_res_feats_0 ──┤
 *                                                      high_res_feats_1 ──┤
 *   Points ──► preprocess_point ──► point_coords ──────────────────────┤
 *                                   point_labels ──────────────────────┤
 *   (none) ──► preprocess_mask ──► mask_input ────────────────────────┤
 *                                   has_mask_input ────────────────────┤
 *   OrigSize ──► orig_im_size ────────────────────────────────────────┤
 *                                                                      ▼
 *                                                              decoder_infer
 *                                                              │
 *                                                              ├──► masks
 *                                                              ├──►
 * iou_predictions └──► low_res_masks │ postprocess │ result
 */
class NNDEPLOY_CC_API SAM2Graph : public dag::Graph {
 public:
  SAM2Graph(const std::string& name) : dag::Graph(name) {
    key_ = "nndeploy::segment::SAM2Graph";
    desc_ = "SAM 2 Graph for image and video segmentation with Hiera backbone.";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<SAM2PointsParam>();
    this->setOutputTypeInfo<cv::Mat>();
    initDynamicsGraphNodes();
    defaultParam();
  }

  SAM2Graph(const std::string& name, std::vector<dag::Edge*> inputs,
            std::vector<dag::Edge*> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::segment::SAM2Graph";
    desc_ = "SAM 2 Graph for image and video segmentation with Hiera backbone.";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<SAM2PointsParam>();
    this->setOutputTypeInfo<cv::Mat>();
    initDynamicsGraphNodes();
    defaultParam();
  }

  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::vector<std::string>& model_value);

  base::Status defaultParam() override;

  /**
   * SAM 2 前向推理
   * inputs[0]: 图像 (cv::Mat)
   * inputs[1]: 点提示 (SAM2PointsParam)
   *
   * 与 SAM 1 的关键区别：
   * encoder 输出 3 个特征（image_embeddings + high_res_feats_0/1）
   * 全部传入 decoder
   */
  std::vector<dag::Edge*> forward(std::vector<dag::Edge*> inputs) {
    std::vector<dag::Edge*> encoder_input =
        (*preprocess_image_node_)({inputs[0]});
    std::vector<dag::Edge*> encoder_outputs =
        (*encoder_infer_node_)(encoder_input);

    std::vector<dag::Edge*> point_result =
        (*preprocess_point_node_)({inputs[1]});
    std::vector<dag::Edge*> mask_result = (*preprocess_mask_node_)();

    // SAM 2 decoder 输入:
    //   encoder_outputs: (ONNX 输出顺序: [0]=high_res_feats_0,
    //   [1]=high_res_feats_1, [2]=image_embed) 共 3 个特征 + point_coords +
    //   point_labels + mask_input + has_mask_input 注意：实际 ONNX 解码器只有 7
    //   个输入，不支持 orig_im_size
    std::vector<dag::Edge*> decoder_input = {
        encoder_outputs[2],  // image_embed (ONNX 输出索引 2)
        encoder_outputs[0],  // high_res_feats_0 (ONNX 输出索引 0)
        encoder_outputs[1],  // high_res_feats_1 (ONNX 输出索引 1)
        point_result[0],     // point_coords
        point_result[1],     // point_labels
        mask_result[0],      // mask_input
        mask_result[1],      // has_mask_input
    };
    std::vector<dag::Edge*> decoder_output =
        (*decoder_infer_node_)(decoder_input);

    std::vector<dag::Edge*> postprocess_output =
        (*postprocess_node_)(decoder_output);
    return postprocess_output;
  }

  /**
   * SAM 2 视频序列推理
   * 在帧间维持 memory，实现时序一致的对象跟踪
   *
   * inputs[0]: 当前帧图像 (cv::Mat)
   * inputs[1]: 点/框提示 (SAM2PointsParam)
   * reset_memory: 是否重置视频记忆（新序列时设为 true）
   */
  std::vector<dag::Edge*> forwardVideo(std::vector<dag::Edge*> inputs,
                                       bool reset_memory = false);

 private:
  base::Status initDynamicsGraphNodes();

 private:
  dag::Node* preprocess_image_node_ = nullptr;
  dag::Node* preprocess_point_node_ = nullptr;
  dag::Node* preprocess_mask_node_ = nullptr;

  infer::Infer* encoder_infer_node_ = nullptr;
  inference::InferenceParam encoder_infer_param_;
  infer::Infer* decoder_infer_node_ = nullptr;
  inference::InferenceParam decoder_infer_param_;

  dag::Node* postprocess_node_ = nullptr;
  SAM2MemoryNode* memory_node_ = nullptr;

  // 视频跟踪用动态边缘（forwardVideo 中用于注入存储的掩码）
  dag::Edge* video_mask_input_edge_ = nullptr;
  dag::Edge* video_has_mask_edge_ = nullptr;
};

}  // namespace segment
}  // namespace nndeploy

#endif /* _NNDEPLOY_SEGMENT_SAM2_H_ */
