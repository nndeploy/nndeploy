
#ifndef _NNDEPLOY_SEGMENT_SAM3_H_
#define _NNDEPLOY_SEGMENT_SAM3_H_

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
#include "nndeploy/detect/result.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_pad_norm_trans.h"
#include "nndeploy/preprocess/params.h"

namespace nndeploy {
namespace segment {

// ============================================================================
// SAM 3 Architecture Overview (Real Architecture)
// ============================================================================
//
// SAM 3 is a decoupled detector-tracker architecture with a shared vision
// encoder. 848M parameters total.
//
// Components:
//   1. Perception Encoder (PE) — Shared ViT backbone (~600M params)
//      - 32 layers, 1024 dim, 16 heads
//      - Produces multi-scale visual features for both Detector and Tracker
//
//   2. Detector (DETR-based, ~50M params)
//      - Fusion Encoder: Cross-attention between PE features and prompt tokens
//      - Transformer Decoder: 6 layers, 200 object queries
//      - Presence Token: A specialized learnable query that discriminates
//        between similar prompts (core innovation of SAM 3)
//      - Presence Head: Decouples recognition (what) from localization (where)
//        Each query predicts a binary "presence" score indicating if an
//        object exists, independent of the concept class.
//      - Segmentation Head: Generates high-res masks from decoder outputs
//
//   3. Tracker (SAM 2-based, ~30M params)
//      - Memory Encoder: Encodes previous masks into memory tokens
//      - Memory Attention: Cross-attention decoder to past frames
//      - Mask Decoder: Two-way transformer for per-instance mask propagation
//      - Supports interactive refinement (iterative exemplar addition)
//
//   4. Text Encoder (~150M params, optional)
//      - CLIP-based text encoder for noun phrase encoding
//      - 24 layers, 1024 width, 16 heads
//
// ============================================================================
// ONNX Model Decomposition (for deployment)
// ============================================================================
//   model_value[0] = image_encoder.onnx  — Shared PE / ViT backbone
//   model_value[1] = detector_decoder.onnx — Fusion Encoder + DETR Decoder
//                                            + Presence Head + SegHead
//   model_value[2] = memory_encoder.onnx — Encode masks to memory tokens
//   model_value[3] = mask_decoder.onnx   — SAM 2-style mask decoder (tracker)
//   model_value[4] = text_encoder.onnx   — CLIP text encoder (optional)
//
// ============================================================================

// ==================== Data Structures ====================

/**
 * @brief Per-instance detection result from SAM 3 detector
 *
 * SAM 3 decomposes detection into:
 *   - Presence score: "does an object exist here?" (binary)
 *   - Concept score: "how well does this object match the concept?"
 *   - Final score = presence × concept (product)
 */
struct DetectionInstance {
  int track_id = -1;
  int concept_idx = 0;          // which concept this matches
  std::string concept_name;     // human-readable concept name
  cv::Rect bbox;                // bounding box
  cv::Mat mask;                 // segmentation mask
  float presence_score = 0.0f;  // from Presence Head (binary objectness)
  float concept_score = 0.0f;   // from Concept Matcher
  float final_score = 0.0f;     // presence * concept
  int age = 0;                  // how many frames tracked
  int lost_age = 0;             // frames since last detection
};

/**
 * @brief Memory bank for SAM 3 tracker
 *
 * Maintains a ring buffer of past frame memory tokens for
 * memory attention across frames.
 *
 * In the real SAM 3 architecture, the tracker uses memory attention
 * to cross-attend to up to N past frames' features, enabling
 * consistent tracking across occlusions and appearance changes.
 */
struct MemoryBank {
  std::vector<device::Tensor*> memory_tensors_;  // memory per frame
  std::vector<int> frame_ids_;                   // frame indices
  int max_frames_ = 16;                          // max tracked frames
  int current_stride_ = 1;                       // frame stride

  void add(device::Tensor* mem, int frame_id) {
    if (memory_tensors_.size() >= (size_t)max_frames_) {
      memory_tensors_.erase(memory_tensors_.begin());
      frame_ids_.erase(frame_ids_.begin());
    }
    memory_tensors_.push_back(mem);
    frame_ids_.push_back(frame_id);
  }

  void reset() {
    memory_tensors_.clear();
    frame_ids_.clear();
  }

  size_t size() const { return memory_tensors_.size(); }
};

// ==================== Parameter Classes (Backward Compatible)
// ====================

/**
 * @brief SAM 3 concept prompt parameter class
 *
 * SAM 3 supports text-based concept prompts (e.g. "car", "person", "dog")
 * for unified detection + segmentation + tracking.
 * Each concept is encoded via a CLIP text encoder into an embedding.
 */
class NNDEPLOY_CC_API Sam3ConceptParam : public base::Param {
 public:
  std::vector<std::string> concepts_;  // text concept list
  int num_concepts_ = 0;
  int concept_dim_ = 512;  // CLIP text embedding dimension
  int ori_width_ = 0;
  int ori_height_ = 0;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief SAM 3 language encoder parameter class
 *
 * Configures the CLIP text encoder model for converting concept
 * text descriptions into embedding vectors.
 */
class NNDEPLOY_CC_API Sam3LanguageParam : public base::Param {
 public:
  int max_token_length_ = 77;  // CLIP max token length
  int hidden_dim_ = 512;       // CLIP text hidden dimension
  int num_concepts_ = 0;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief SAM 3 exemplar prompt parameter class
 *
 * SAM 3 supports exemplar (few-shot) prompting: providing example images
 * of a concept to guide segmentation, without needing text descriptions.
 *
 * Each exemplar is a reference image + optional mask of the target object.
 */
class NNDEPLOY_CC_API Sam3ExemplarParam : public base::Param {
 public:
  std::vector<cv::Mat> exemplar_images_;  // reference images
  std::vector<cv::Mat> exemplar_masks_;   // optional reference masks
  int num_exemplars_ = 0;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief SAM 3 post-processing parameter class
 *
 * SAM 3 decoder outputs masks, presence scores, and per-concept similarities.
 * Post-processing selects the best matching instances and produces
 * a labeled segmentation map with track IDs.
 */
class NNDEPLOY_CC_API Sam3PostParam : public base::Param {
 public:
  float score_threshold_ = 0.5f;
  float presence_threshold_ = 0.3f;  // minimum presence score
  float concept_threshold_ = 0.3f;   // minimum concept similarity
  int model_h_ = 1024;
  int model_w_ = 1024;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief Tracked instance across video frames (kept for backward compat)
 */
struct TrackInstance {
  int track_id = -1;
  int concept_idx = 0;
  cv::Rect bbox;
  cv::Mat mask;
  float score = 0.0f;
  int age = 0;
  int lost_age = 0;
};

/**
 * @brief SAM 3 video tracking parameter class
 */
// ==================== New Architecture Parameter Classes ====================

/**
 * @brief Perception Encoder parameters
 *
 * Shared ViT backbone used by both Detector and Tracker.
 * Same architecture as SAM 1/2 image encoder.
 */
class NNDEPLOY_CC_API Sam3PerceptionEncoderParam : public base::Param {
 public:
  int embed_dim_ = 256;
  int image_size_ = 1024;
  bool use_high_res_ = false;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief DETR Detector Decoder parameters
 *
 * Configures the DETR-based detector decoder that takes
 * PE features + concept embeddings and produces per-query masks,
 * presence scores, and box predictions.
 *
 * In SAM 3, this includes:
 *   - Fusion Encoder (cross-attend PE features with prompt tokens)
 *   - Transformer Decoder (6 layers, 200 queries)
 *   - Presence Token (specialized discriminative query)
 *   - Segmentation Head (high-res mask generation)
 */
class NNDEPLOY_CC_API Sam3DetectorDecoderParam : public base::Param {
 public:
  int num_queries_ = 200;           // DETR object queries
  int query_dim_ = 256;             // query embedding dimension
  bool has_presence_token_ = true;  // SAM 3 presence token enabled
  int num_decoder_layers_ = 6;      // number of decoder layers
  bool box_refine_ = true;          // iterative box refinement

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief Presence Head parameters
 *
 * Presence Head is the core architectural innovation of SAM 3.
 * It decouples recognition (what object) from localization (where object):
 *
 *   presence_score = sigmoid(MLP(query_embedding))
 *
 * A high presence score means "there is an object here" regardless
 * of what concept it belongs to.
 *
 * The final detection score = presence_score * concept_similarity
 */
class NNDEPLOY_CC_API Sam3PresenceHeadParam : public base::Param {
 public:
  float presence_threshold_ = 0.3f;  // binary presence cutoff
  bool enable_nms_ = true;           // apply NMS after presence filtering
  float nms_threshold_ = 0.7f;       // NMS IOU threshold

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief Concept Matcher parameters
 *
 * Matches detected objects (from Detector Decoder + Presence Head)
 * to text/exemplar concept embeddings using dot-product similarity.
 *
 * In SAM 3, concept matching is:
 *   concept_score = softmax(query_embedding @ concept_embedding^T)
 *
 * The object is assigned to the concept with highest similarity score.
 */
class NNDEPLOY_CC_API Sam3ConceptMatcherParam : public base::Param {
 public:
  float similarity_threshold_ = 0.2f;         // min dot-product similarity
  std::string scoring_type_ = "dot_product";  // dot_product | cosine
  bool per_concept_nms_ = true;               // apply NMS per concept class

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief Memory Encoder parameters
 *
 * Encodes per-instance masks into compact memory tokens that
 * are stored in the MemoryBank for cross-frame attention.
 */
class NNDEPLOY_CC_API Sam3MemoryEncoderParam : public base::Param {
 public:
  int memory_dim_ = 256;
  int max_memory_frames_ = 16;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief Tracker Mask Decoder parameters
 *
 * SAM 2-style mask decoder for temporal mask propagation.
 * Uses memory attention to cross-attend past frame features
 * and produce refined mask predictions for the current frame.
 */
class NNDEPLOY_CC_API Sam3TrackerMaskDecoderParam : public base::Param {
 public:
  int num_mask_embeddings_ = 4;  // number of mask input embeddings
  int embedding_dim_ = 256;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief Memory Manager parameters
 *
 * Manages the memory state across video frames.
 * Controls the ring buffer size, frame stride, and eviction policy.
 */
class NNDEPLOY_CC_API Sam3MemoryManagerParam : public base::Param {
 public:
  int max_memory_frames_ = 16;
  int frame_stride_ = 1;
  bool enable_memory_temperature_ = false;  // weight older frames less

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

// ==================== Legacy Node Classes (Backward Compatible)
// ==================== NOTE: These legacy nodes are kept for backward
// compatibility. For new code, use the simplified Sam3Simple* nodes that
// directly map to ONNX models from vietanhdev/samexporter / wkentaro/sam3-onnx.

/**
 * @brief SAM 3 Language Encoder Node (legacy, kept for backward compat)
 *
 * Encodes text concept descriptions into embedding vectors using a
 * CLIP text encoder ONNX model.
 *
 * In the new architecture, this feeds into the DetectorDecoder's
 * cross-attention (Fusion Encoder).
 */
class NNDEPLOY_CC_API Sam3LanguageEncodeNode : public dag::Node {
 public:
  Sam3LanguageEncodeNode(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3LanguageEncodeNode";
    desc_ = "SAM 3 Language Encode Node - CLIP text encoder.";
    param_ = std::make_shared<Sam3LanguageParam>();
    this->defaultParam();
  }
  Sam3LanguageEncodeNode(const std::string& name,
                         std::vector<dag::Edge*> inputs,
                         std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3LanguageEncodeNode";
    desc_ = "SAM 3 Language Encode Node - CLIP text encoder.";
    param_ = std::make_shared<Sam3LanguageParam>();
    this->defaultParam();
  }

  virtual ~Sam3LanguageEncodeNode() {}

  virtual base::Status run();

  base::Status setTextEncoderInferParam(base::InferenceType inference_type,
                                        base::DeviceType device_type,
                                        base::ModelType model_type,
                                        bool is_path, std::string& model_path);

  base::Status defaultParam() override;

 private:
  infer::Infer* text_encoder_infer_ = nullptr;
  inference::InferenceParam text_encoder_param_;
  bool has_text_encoder_ = false;
};

/**
 * @brief SAM 3 Concept Encode Node (legacy, kept for backward compatibility)
 *
 * Creates placeholder concept embeddings when no language encoder is used.
 */
class NNDEPLOY_CC_API Sam3ConceptEncodeNode : public dag::Node {
 public:
  Sam3ConceptEncodeNode(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3ConceptEncodeNode";
    desc_ = "SAM 3 Concept Encode Node for concept prompt encoding.";
    this->setInputTypeInfo<Sam3ConceptParam>();
    this->setOutputTypeInfo<device::Tensor>();
    param_ = std::make_shared<Sam3ConceptParam>();
    this->defaultParam();
  }
  Sam3ConceptEncodeNode(const std::string& name, std::vector<dag::Edge*> inputs,
                        std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3ConceptEncodeNode";
    desc_ = "SAM 3 Concept Encode Node for concept prompt encoding.";
    this->setInputTypeInfo<Sam3ConceptParam>();
    this->setOutputTypeInfo<device::Tensor>();
    param_ = std::make_shared<Sam3ConceptParam>();
    this->defaultParam();
  }
  virtual ~Sam3ConceptEncodeNode() {}

  virtual base::Status run();

  base::Status setConcepts(const std::vector<std::string>& concepts) {
    Sam3ConceptParam* param = dynamic_cast<Sam3ConceptParam*>(param_.get());
    if (concepts.empty()) {
      NNDEPLOY_LOGE("Concepts list is empty.");
      return base::kStatusCodeErrorInvalidValue;
    }
    param->concepts_ = concepts;
    param->num_concepts_ = concepts.size();
    return base::kStatusCodeOk;
  }

  base::Status defaultParam() override {
    Sam3ConceptParam* param = dynamic_cast<Sam3ConceptParam*>(param_.get());
    param->concepts_.clear();
    param->num_concepts_ = 1;
    param->concept_dim_ = 512;
    param->ori_width_ = 0;
    param->ori_height_ = 0;
    return base::kStatusCodeOk;
  }
};

/**
 * @brief SAM 3 Exemplar Encode Node
 *
 * Encodes exemplar (reference) images into embedding vectors using the
 * image encoder. Used for few-shot prompting where text is insufficient.
 *
 * In the new architecture, exemplar embeddings are passed alongside
 * text embeddings to the DetectorDecoder's Fusion Encoder.
 */
class NNDEPLOY_CC_API Sam3ExemplarEncodeNode : public dag::Node {
 public:
  Sam3ExemplarEncodeNode(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3ExemplarEncodeNode";
    desc_ = "SAM 3 Exemplar Encode Node.";
    param_ = std::make_shared<Sam3ExemplarParam>();
    this->defaultParam();
  }
  Sam3ExemplarEncodeNode(const std::string& name,
                         std::vector<dag::Edge*> inputs,
                         std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3ExemplarEncodeNode";
    desc_ = "SAM 3 Exemplar Encode Node.";
    param_ = std::make_shared<Sam3ExemplarParam>();
    this->defaultParam();
  }
  virtual ~Sam3ExemplarEncodeNode() {}

  virtual base::Status run();
  base::Status defaultParam() override;

  // Set the inference param to point to the shared PE model
  base::Status setSharedEncoder(infer::Infer* pe_infer);
  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::string& model_path);

 private:
  infer::Infer* exemplar_encoder_infer_ = nullptr;
  inference::InferenceParam exemplar_encoder_param_;
  bool has_encoder_ = false;
};

/**
 * @brief SAM 3 Post-Process Node (Updated for new architecture)
 *
 * Converts decoder output (masks + presence scores + concept matches)
 * into a labeled segmentation visualization with per-instance tracking.
 *
 * Inputs:
 *   [0] masks tensor (batch, num_masks, H, W)
 *   [1] presence scores (batch, num_masks) — from Presence Head
 *   [2] concept matches (batch, num_masks, num_concepts) — from Concept Matcher
 * Output:
 *   [0] cv::Mat - color segmentation map with track IDs
 *   [1] std::vector<DetectionInstance> (optional) - per-instance results
 */
class NNDEPLOY_CC_API Sam3PostProcess : public dag::Node {
 public:
  Sam3PostProcess(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3PostProcess";
    desc_ = "SAM 3 Post Process Node with presence + concept matching.";
    param_ = std::make_shared<Sam3PostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  Sam3PostProcess(const std::string& name, std::vector<dag::Edge*> inputs,
                  std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3PostProcess";
    desc_ = "SAM 3 Post Process Node with presence + concept matching.";
    param_ = std::make_shared<Sam3PostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~Sam3PostProcess() {}

  virtual base::Status run();
  base::Status defaultParam() override;
};

// ==================== NEW: Real SAM 3 Architecture Nodes ====================

/**
 * @brief SAM 3 Perception Encoder Node
 *
 * Shared ViT backbone that produces multi-scale visual features.
 * Used by both the Detector and Tracker.
 *
 * This is the same underlying model as "image_encoder" in SAM 1/2,
 * but renamed to match SAM 3's "Perception Encoder" terminology.
 *
 * Inputs:
 *   [0] preprocessed image tensor (1, 3, 1024, 1024)
 * Outputs:
 *   [0] image features (1, 256, 64, 64) — shared representation
 */
class NNDEPLOY_CC_API Sam3PerceptionEncoder : public dag::Node {
 public:
  Sam3PerceptionEncoder(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3PerceptionEncoder";
    desc_ = "SAM 3 Perception Encoder (shared ViT backbone).";
    param_ = std::make_shared<Sam3PerceptionEncoderParam>();
    this->defaultParam();
  }
  Sam3PerceptionEncoder(const std::string& name, std::vector<dag::Edge*> inputs,
                        std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3PerceptionEncoder";
    desc_ = "SAM 3 Perception Encoder (shared ViT backbone).";
    param_ = std::make_shared<Sam3PerceptionEncoderParam>();
    this->defaultParam();
  }
  virtual ~Sam3PerceptionEncoder() {}

  virtual base::Status run();
  base::Status defaultParam() override;

  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::string& model_path);

  // Get the internal infer node (for sharing with ExemplarEncodeNode)
  infer::Infer* getInferNode() { return pe_infer_node_; }

 private:
  infer::Infer* pe_infer_node_ = nullptr;
  inference::InferenceParam pe_infer_param_;
};

/**
 * @brief SAM 3 DETR Detector Decoder Node
 *
 * The core detection component that takes PE features + concept embeddings
 * and produces per-query mask predictions, presence scores, and boxes.
 *
 * Architecture (all fused in one ONNX model):
 *   1. Fusion Encoder: Cross-attention between PE features and prompt tokens
 *   2. Transformer Decoder: 6 layers, 200 object queries, with Presence Token
 *   3. Segmentation Head: Generate high-res masks from query embeddings
 *
 * Inputs:
 *   [0] image features (1, 256, 64, 64) — from Perception Encoder
 *   [1] concept embeddings (1, num_concepts, 512) — from Text/Exemplar Encoder
 * Outputs:
 *   [0] masks tensor (1, num_queries, H, W) — per-query segmentation masks
 *   [1] presence scores (1, num_queries) — binary presence per query
 *   [2] refined query embeddings (1, num_queries, dim) — for concept matching
 *   [3] box predictions (1, num_queries, 4) — optional bounding boxes
 */
class NNDEPLOY_CC_API Sam3DetectorDecoder : public dag::Node {
 public:
  Sam3DetectorDecoder(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3DetectorDecoder";
    desc_ = "SAM 3 DETR Detector Decoder with Presence Token.";
    param_ = std::make_shared<Sam3DetectorDecoderParam>();
    this->defaultParam();
  }
  Sam3DetectorDecoder(const std::string& name, std::vector<dag::Edge*> inputs,
                      std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3DetectorDecoder";
    desc_ = "SAM 3 DETR Detector Decoder with Presence Token.";
    param_ = std::make_shared<Sam3DetectorDecoderParam>();
    this->defaultParam();
  }
  virtual ~Sam3DetectorDecoder() {}

  virtual base::Status run();
  base::Status defaultParam() override;

  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::string& model_path);

 private:
  infer::Infer* detector_infer_node_ = nullptr;
  inference::InferenceParam detector_infer_param_;
};

/**
 * @brief SAM 3 Presence Head Node
 *
 * **Core architectural innovation of SAM 3.**
 *
 * Processes per-query presence scores from the DetectorDecoder and applies
 * thresholding and Non-Maximum Suppression (NMS) to produce filtered
 * detections. The Presence Head decouples recognition from localization:
 *
 * - Recognition is the process of matching object queries with concept
 *   embeddings to assign semantic labels.
 * - Localization is the process of determining WHERE objects are in the
 *   image, regardless of what they are.
 *
 * This decoupling allows SAM 3 to:
 *   1. Discriminate between similar prompts (e.g. "player in white" vs
 *      "player in red") via the Presence Token
 *   2. Handle negative phrases effectively
 *   3. Work with open-vocabulary concepts
 *
 * Inputs:
 *   [0] presence scores (1, num_queries) — from DetectorDecoder
 *   [1] masks tensor (1, num_queries, H, W) — for NMS
 *   [2] query embeddings (1, num_queries, dim) — for concept matching
 * Outputs:
 *   [0] filtered query indices (1, num_filtered) — surviving queries
 *   [1] filtered masks (1, num_filtered, H, W)
 *   [2] filtered embeddings (1, num_filtered, dim)
 *   [3] filtered presence scores (1, num_filtered)
 */
class NNDEPLOY_CC_API Sam3PresenceHead : public dag::Node {
 public:
  Sam3PresenceHead(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3PresenceHead";
    desc_ = "SAM 3 Presence Head: decoupled recognition & localization.";
    param_ = std::make_shared<Sam3PresenceHeadParam>();
    this->defaultParam();
  }
  Sam3PresenceHead(const std::string& name, std::vector<dag::Edge*> inputs,
                   std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3PresenceHead";
    desc_ = "SAM 3 Presence Head: decoupled recognition & localization.";
    param_ = std::make_shared<Sam3PresenceHeadParam>();
    this->defaultParam();
  }
  virtual ~Sam3PresenceHead() {}

  virtual base::Status run();
  base::Status defaultParam() override;

 private:
  // Compute pairwise mask IOU for NMS
  float computeMaskIOU(const float* mask_a, const float* mask_b, int h, int w);
};

/**
 * @brief SAM 3 Concept Matcher Node
 *
 * Matches surviving detection queries (from Presence Head) to concept
 * embeddings using dot-product / cosine similarity.
 *
 * In the real SAM 3 architecture, the final score for each detection is:
 *   final_score = presence_score × concept_similarity
 *
 * Where:
 *   - presence_score = binary objectness (from Presence Head)
 *   - concept_similarity = softmax(query_emb @ concept_emb^T)
 *
 * Inputs:
 *   [0] filtered query embeddings (1, num_filtered, dim) — from PresenceHead
 *   [1] concept embeddings (1, num_concepts, dim) — from Text/Exemplar Encoder
 * Outputs:
 *   [0] concept scores (1, num_filtered, num_concepts) — per-query, per-concept
 *   [1] final scores (1, num_filtered) — presence × concept product
 *   [2] best concept indices (1, num_filtered) — argmax per query
 */
class NNDEPLOY_CC_API Sam3ConceptMatcher : public dag::Node {
 public:
  Sam3ConceptMatcher(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3ConceptMatcher";
    desc_ = "SAM 3 Concept Matcher: match queries to concepts.";
    param_ = std::make_shared<Sam3ConceptMatcherParam>();
    this->defaultParam();
  }
  Sam3ConceptMatcher(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3ConceptMatcher";
    desc_ = "SAM 3 Concept Matcher: match queries to concepts.";
    param_ = std::make_shared<Sam3ConceptMatcherParam>();
    this->defaultParam();
  }
  virtual ~Sam3ConceptMatcher() {}

  virtual base::Status run();
  base::Status defaultParam() override;
};

/**
 * @brief SAM 3 Memory Encoder Node
 *
 * Encodes per-instance mask predictions into compact memory tokens
 * that are stored in the MemoryBank for future frame attention.
 *
 * This enables the tracker to maintain temporal consistency by
 * cross-attending to past frame features.
 *
 * Inputs:
 *   [0] image features (1, 256, 64, 64) — from Perception Encoder
 *   [1] mask predictions (1, num_instances, H, W) — from tracker
 * Outputs:
 *   [0] memory tokens (1, num_instances, memory_dim) — per-instance memory
 */
class NNDEPLOY_CC_API Sam3MemoryEncoder : public dag::Node {
 public:
  Sam3MemoryEncoder(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3MemoryEncoder";
    desc_ = "SAM 3 Memory Encoder.";
    param_ = std::make_shared<Sam3MemoryEncoderParam>();
    this->defaultParam();
  }
  Sam3MemoryEncoder(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3MemoryEncoder";
    desc_ = "SAM 3 Memory Encoder.";
    param_ = std::make_shared<Sam3MemoryEncoderParam>();
    this->defaultParam();
  }
  virtual ~Sam3MemoryEncoder() {}

  virtual base::Status run();
  base::Status defaultParam() override;

  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::string& model_path);

 private:
  infer::Infer* memory_infer_node_ = nullptr;
  inference::InferenceParam memory_infer_param_;
};

/**
 * @brief SAM 3 Tracker Mask Decoder Node
 *
 * SAM 2-style mask decoder for temporal mask propagation across video frames.
 * Uses memory attention to cross-attend past frame features and produce
 * refined mask predictions for the current frame.
 *
 * Architecture follows SAM 2 tracker:
 *   - Memory Attention: Cross-attend current features to memory bank
 *   - Mask Decoder: Two-way transformer for per-instance mask prediction
 *
 * Inputs:
 *   [0] image features (1, 256, 64, 64) — from Perception Encoder
 *   [1] memory bank features (1, num_memory_frames * dim) — concatenated
 *   [2] object query embeddings (1, num_instances, dim)
 * Outputs:
 *   [0] refined masks (1, num_instances, H, W)
 *   [1] per-instance scores (1, num_instances)
 */
class NNDEPLOY_CC_API Sam3TrackerMaskDecoder : public dag::Node {
 public:
  Sam3TrackerMaskDecoder(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3TrackerMaskDecoder";
    desc_ = "SAM 3 Tracker Mask Decoder (SAM 2-style).";
    param_ = std::make_shared<Sam3TrackerMaskDecoderParam>();
    this->defaultParam();
  }
  Sam3TrackerMaskDecoder(const std::string& name,
                         std::vector<dag::Edge*> inputs,
                         std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3TrackerMaskDecoder";
    desc_ = "SAM 3 Tracker Mask Decoder (SAM 2-style).";
    param_ = std::make_shared<Sam3TrackerMaskDecoderParam>();
    this->defaultParam();
  }
  virtual ~Sam3TrackerMaskDecoder() {}

  virtual base::Status run();
  base::Status defaultParam() override;

  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::string& model_path);

 private:
  infer::Infer* tracker_infer_node_ = nullptr;
  inference::InferenceParam tracker_infer_param_;
};

/**
 * @brief SAM 3 Memory Manager Node
 *
 * Manages the cross-frame memory state for the tracker.
 * Maintains a ring buffer (MemoryBank) of past frame memory tokens,
 * handling insertion, eviction, and concatenation for model input.
 *
 * The Memory Manager is a C++ node (no ONNX model) that orchestrates
 * the memory state across frames.
 *
 * Inputs:
 *   [0] new memory tokens (1, num_instances, memory_dim) — from MemoryEncoder
 *   [1] frame index (int32) — current frame number
 * Outputs:
 *   [0] concatenated memory bank (1, num_memory_frames * memory_dim)
 *   [1] memory valid mask (1, num_memory_frames) — which slots are valid
 */
class NNDEPLOY_CC_API Sam3MemoryManager : public dag::Node {
 public:
  Sam3MemoryManager(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3MemoryManager";
    desc_ = "SAM 3 Memory Manager.";
    param_ = std::make_shared<Sam3MemoryManagerParam>();
    this->defaultParam();
  }
  Sam3MemoryManager(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3MemoryManager";
    desc_ = "SAM 3 Memory Manager.";
    param_ = std::make_shared<Sam3MemoryManagerParam>();
    this->defaultParam();
  }
  virtual ~Sam3MemoryManager() {}

  virtual base::Status run();
  base::Status defaultParam() override;

  void reset() {
    memory_bank_.reset();
    current_frame_ = 0;
  }

 private:
  MemoryBank memory_bank_;
  int current_frame_ = 0;
};

// ==================== Simplified ONNX Node Parameter Classes
// ====================

/**
 * @brief SAM 3 simplified Image Encoder parameter class
 *
 * Maps directly to sam3_image_encoder.onnx.
 * Input: image tensor (3, 1008, 1008) float32
 * Output: 6 tensors — vision_pos_enc[0:3] + backbone_fpn[0:3]
 */
class NNDEPLOY_CC_API Sam3SimpleImageEncoderParam : public base::Param {
 public:
  int image_height_ = 1008;
  int image_width_ = 1008;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief SAM 3 simplified Language Encoder parameter class
 *
 * Maps directly to sam3_language_encoder.onnx.
 * Input: tokens tensor (1, 32) int64
 * Output: 3 tensors — language_mask, language_features, language_embeds
 */
class NNDEPLOY_CC_API Sam3SimpleLanguageEncoderParam : public base::Param {
 public:
  int max_token_length_ = 32;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief SAM 3 simplified Decoder parameter class
 *
 * Maps directly to sam3_decoder.onnx.
 * Inputs: backbone_fpn[0:3], vision_pos_enc_2, language_mask,
 *         language_features, box_coords, box_labels, box_masks
 * Output: 3 tensors — boxes, scores, masks
 */
class NNDEPLOY_CC_API Sam3SimpleDecoderParam : public base::Param {
 public:
  float score_threshold_ = 0.5f;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

// ==================== Simplified ONNX Node Classes ====================

/**
 * @brief SAM 3 simplified Image Encoder Node
 *
 * Direct ONNX wrapper for sam3_image_encoder.onnx.
 * Input:  [0] image tensor (3, 1008, 1008) float32 — preprocessed CHW
 * Output: [0] vision_pos_enc_0
 *         [1] vision_pos_enc_1
 *         [2] vision_pos_enc_2
 *         [3] backbone_fpn_0
 *         [4] backbone_fpn_1
 *         [5] backbone_fpn_2
 */
class NNDEPLOY_CC_API Sam3SimpleImageEncoder : public dag::Node {
 public:
   Sam3SimpleImageEncoder(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3SimpleImageEncoder";
    desc_ = "SAM 3 Simplified Image Encoder (direct ONNX).";
    param_ = std::make_shared<Sam3SimpleImageEncoderParam>();
    this->defaultParam();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  Sam3SimpleImageEncoder(const std::string& name,
                         std::vector<dag::Edge*> inputs,
                         std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3SimpleImageEncoder";
    desc_ = "SAM 3 Simplified Image Encoder (direct ONNX).";
    param_ = std::make_shared<Sam3SimpleImageEncoderParam>();
    this->defaultParam();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~Sam3SimpleImageEncoder() {}

  virtual base::Status run();
  base::Status defaultParam() override;

  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::string& model_path);

  // Set external model data path for ONNX models with external data
  base::Status setExternalModelData(const std::string& external_data_path) {
    if (image_encoder_infer_ != nullptr) {
      auto param = dynamic_cast<inference::InferenceParam*>(
          image_encoder_infer_->getParam());
      if (param != nullptr) {
        param->external_model_data_ = {external_data_path};
      }
    }
    return base::kStatusCodeOk;
  }

  // Get internal infer node for initialization
  infer::Infer* getInferNode() { return image_encoder_infer_; }

 private:
  infer::Infer* image_encoder_infer_ = nullptr;
  inference::InferenceParam image_encoder_param_;
};

/**
 * @brief SAM 3 simplified Language Encoder Node
 *
 * Direct ONNX wrapper for sam3_language_encoder.onnx.
 * Input:  [0] tokens tensor (1, 32) int64 — tokenized text prompt
 * Output: [0] language_mask (1, 32) bool
 *         [1] language_features (1, 32, 768) float32
 *         [2] language_embeds (1, 768) float32
 */
class NNDEPLOY_CC_API Sam3SimpleLanguageEncoder : public dag::Node {
 public:
  Sam3SimpleLanguageEncoder(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3SimpleLanguageEncoder";
    desc_ = "SAM 3 Simplified Language Encoder (direct ONNX).";
    param_ = std::make_shared<Sam3SimpleLanguageEncoderParam>();
    this->defaultParam();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  Sam3SimpleLanguageEncoder(const std::string& name,
                            std::vector<dag::Edge*> inputs,
                            std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3SimpleLanguageEncoder";
    desc_ = "SAM 3 Simplified Language Encoder (direct ONNX).";
    param_ = std::make_shared<Sam3SimpleLanguageEncoderParam>();
    this->defaultParam();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~Sam3SimpleLanguageEncoder() {}

  virtual base::Status run();
  base::Status defaultParam() override;

  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::string& model_path);

  // Set external model data path for ONNX models with external data
  base::Status setExternalModelData(const std::string& external_data_path) {
    if (language_encoder_infer_ != nullptr) {
      auto param = dynamic_cast<inference::InferenceParam*>(
          language_encoder_infer_->getParam());
      if (param != nullptr) {
        param->external_model_data_ = {external_data_path};
      }
    }
    return base::kStatusCodeOk;
  }

  // Get internal infer node for initialization
  infer::Infer* getInferNode() { return language_encoder_infer_; }

 private:
  infer::Infer* language_encoder_infer_ = nullptr;
  inference::InferenceParam language_encoder_param_;
};

/**
 * @brief SAM 3 simplified Decoder Node
 *
 * Direct ONNX wrapper for sam3_decoder.onnx (11 inputs, 3 outputs).
 * Inputs (ONNX order — must match setInputName/setInputName bindings):
 *   [0]  original_height (int64 scalar)   — input image height in pixels
 *   [1]  original_width  (int64 scalar)   — input image width in pixels
 *   [2]  vision_pos_enc_2 (1, 256, 63, 63) — positional encoding (min scale)
 *   [3]  backbone_fpn_0  (1, 256, 252, 252) — FPN feature 0 (high res)
 *   [4]  backbone_fpn_1  (1, 256, 126, 126) — FPN feature 1 (mid res)
 *   [5]  backbone_fpn_2  (1, 256, 63, 63)   — FPN feature 2 (low res)
 *   [6]  language_mask   (1, 32) bool        — text attention mask
 *   [7]  language_features (32, 1, 256)      — text memory features
 *   [8]  box_coords      (1, 1, 4) float32   — normalized [cx,cy,w,h]
 *   [9]  box_labels      (1, 1) int64        — 1=positive, 0=negative,
 * -1=ignore [10] box_masks       (1, 1) bool         — true=use box prompt
 * Outputs:
 *   [0] boxes  (N, 4)        — predicted bounding boxes (normalized)
 *   [1] scores (N,) float32  — confidence scores
 *   [2] masks  (N, 1, H, W) bool — segmentation masks
 */
class NNDEPLOY_CC_API Sam3SimpleDecoder : public dag::Node {
 public:
   Sam3SimpleDecoder(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3SimpleDecoder";
    desc_ = "SAM 3 Simplified Decoder (direct ONNX).";
    param_ = std::make_shared<Sam3SimpleDecoderParam>();
    this->defaultParam();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  Sam3SimpleDecoder(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3SimpleDecoder";
    desc_ = "SAM 3 Simplified Decoder (direct ONNX).";
    param_ = std::make_shared<Sam3SimpleDecoderParam>();
    this->defaultParam();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~Sam3SimpleDecoder() {}

  virtual base::Status run();
  base::Status defaultParam() override;

  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::string& model_path);

  // Set external model data path for ONNX models with external data
  base::Status setExternalModelData(const std::string& external_data_path) {
    if (decoder_infer_ != nullptr) {
      auto param =
          dynamic_cast<inference::InferenceParam*>(decoder_infer_->getParam());
      if (param != nullptr) {
        param->external_model_data_ = {external_data_path};
      }
    }
    return base::kStatusCodeOk;
  }

  // Get internal infer node for initialization
  infer::Infer* getInferNode() { return decoder_infer_; }

 private:
  infer::Infer* decoder_infer_ = nullptr;
  inference::InferenceParam decoder_param_;
};

// ==================== Simplified PostProcess Node ====================

/**
 * @brief SAM 3 simplified PostProcess node
 *
 * Filters decoder outputs (boxes, scores, masks) by score threshold
 * and creates a color-coded segmentation visualization.
 *
 * Inputs:
 *   [0] boxes (N, 4) float32 — normalized bounding boxes
 *   [1] scores (N,) float32  — confidence scores
 *   [2] masks (N, 1, H, W) float32 — segmentation masks
 * Output:
 *   [0] cv::Mat — color segmentation visualization
 */
class NNDEPLOY_CC_API Sam3SimplePostprocess : public dag::Node {
 public:
  Sam3SimplePostprocess(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::segment::Sam3SimplePostprocess";
    desc_ = "SAM 3 Simplified PostProcess (score filter + visualization).";
    param_ = std::make_shared<Sam3PostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  Sam3SimplePostprocess(const std::string& name, std::vector<dag::Edge*> inputs,
                        std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3SimplePostprocess";
    desc_ = "SAM 3 Simplified PostProcess (score filter + visualization).";
    param_ = std::make_shared<Sam3PostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~Sam3SimplePostprocess() {}

  virtual base::Status run();
  base::Status defaultParam() override;
};

// ==================== SAM 3 Simplified Graph ====================

/**
 * @brief SAM 3 Simplified Graph Parameter
 *
 * Stores model paths for the 3 simplified ONNX models:
 *   [0] = sam3_image_encoder.onnx
 *   [1] = sam3_decoder.onnx
 *   [2] = sam3_language_encoder.onnx (optional)
 */
class NNDEPLOY_CC_API Sam3SimpleGraphParam : public base::Param {
 public:
  std::string inference_type_ = "kInferenceTypeOnnxRuntime";
  std::string device_type_ = "kDeviceTypeCodeCpu:0";
  std::string model_type_ = "kModelTypeOnnx";
  bool is_path_ = true;
  std::vector<std::string> model_value_;
  std::vector<std::string>
      external_model_data_;  // External data files for ONNX models
  std::string text_prompt_;  // Text prompt (e.g. "person")
  std::vector<int64_t>
      token_ids_;  // Pre-computed CLIP token IDs (context_length=32)

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief SAM 3 Simplified Graph
 *
 * A minimal pipeline using the 3 Sam3Simple* ONNX nodes directly:
 *
 *   Image ──► Preprocess ──► Sam3SimpleImageEncoder ──┐
 *                                                     │
 *   Text Tokens ──► Sam3SimpleLanguageEncoder ──┐     │
 *                                               │     │
 *                                               ▼     ▼
 *                                         Sam3SimpleDecoder
 *                                               │
 *                                               ▼
 *                                        Sam3SimplePostprocess
 *                                               │
 *                                               ▼
 *                                            cv::Mat
 *
 * Supports text-only and text+box prompts.
 * Keeps existing SAM3Graph for backward compatibility.
 */
class NNDEPLOY_CC_API Sam3SimpleGraph : public dag::Graph {
 public:
  Sam3SimpleGraph(const std::string& name) : dag::Graph(name) {
    key_ = "nndeploy::segment::Sam3SimpleGraph";
    desc_ = "SAM 3 Simplified Graph: 3-node ONNX pipeline.";
    param_ = std::make_shared<Sam3SimpleGraphParam>();
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<std::vector<int64_t>>();
    this->setOutputTypeInfo<cv::Mat>();
    initDynamicsGraphNodes();
    defaultParam();
  }

  Sam3SimpleGraph(const std::string& name, std::vector<dag::Edge*> inputs,
                  std::vector<dag::Edge*> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::segment::Sam3SimpleGraph";
    desc_ = "SAM 3 Simplified Graph: 3-node ONNX pipeline.";
    param_ = std::make_shared<Sam3SimpleGraphParam>();
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<std::vector<int64_t>>();
    this->setOutputTypeInfo<cv::Mat>();
    initDynamicsGraphNodes();
    defaultParam();
  }

  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::vector<std::string>& model_value,
                             std::vector<std::string>& external_model_data);

  base::Status defaultParam() override;

  base::Status init() override;

  base::Status run() override;

  using dag::Graph::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;

  std::vector<dag::Edge*> forward(std::vector<dag::Edge*> inputs) override;

 private:
  base::Status initDynamicsGraphNodes();

  // Internal nodes
  preprocess::CvtResizePadNormTrans* preprocess_node_ = nullptr;
  Sam3SimpleImageEncoder* image_encoder_node_ = nullptr;
  Sam3SimpleLanguageEncoder* language_encoder_node_ = nullptr;
  Sam3SimpleDecoder* decoder_node_ = nullptr;
  Sam3SimplePostprocess* postprocess_node_ = nullptr;

  // Dynamic edges for decoder box prompts (created in initDynamicsGraphNodes)
  dag::Edge* box_coords_edge_ = nullptr;
  dag::Edge* box_labels_edge_ = nullptr;
  dag::Edge* box_masks_edge_ = nullptr;

  // Dynamic edges for original image dimensions (11th and 12th decoder inputs)
  dag::Edge* original_height_edge_ = nullptr;
  dag::Edge* original_width_edge_ = nullptr;

  // Optional box prompt inputs
  dag::Edge* input_box_coords_ = nullptr;
  dag::Edge* input_box_labels_ = nullptr;
  dag::Edge* input_box_masks_ = nullptr;
};

// ==================== SAM 3 Graph Param (for JSON deserialization)
// ====================

/**
 * @brief SAM 3 Graph parameter class for JSON workflow deserialization
 *
 * Stores model paths and inference configuration so that SAM3Graph can
 * be fully configured from a JSON workflow file without C++ API calls.
 *
 * JSON example:
 * @code{.json}
 * "param_": {
 *     "inference_type_": "kInferenceTypeOnnxRuntime",
 *     "device_type_": "kDeviceTypeCodeCpu:0",
 *     "model_type_": "kModelTypeOnnx",
 *     "is_path_": true,
 *     "model_value_": [
 *         "path/to/sam3_image_encoder.onnx",
 *         "path/to/sam3_decoder.onnx",
 *         "path/to/memory_encoder.onnx",
 *         "path/to/tracker_decoder.onnx",
 *         "path/to/sam3_language_encoder.onnx"
 *     ]
 * }
 * @endcode
 */
class NNDEPLOY_CC_API Sam3GraphParam : public base::Param {
 public:
  /** Inference backend type (e.g. kInferenceTypeOnnxRuntime) */
  std::string inference_type_ = "kInferenceTypeOnnxRuntime";
  /** Device type (e.g. kDeviceTypeCodeCpu:0) */
  std::string device_type_ = "kDeviceTypeCodeCpu:0";
  /** Model format (e.g. kModelTypeOnnx) */
  std::string model_type_ = "kModelTypeOnnx";
  /** Whether model_value_ contains file paths (true) or model data (false) */
  bool is_path_ = true;
  /**
   * Model file paths / identifiers, indexed by role:
   *   [0] = image_encoder.onnx  — Perception Encoder (shared ViT)
   *   [1] = decoder.onnx        — DETR Detector Decoder
   *   [2] = memory_encoder.onnx — Memory Encoder (video tracking)
   *   [3] = tracker_decoder.onnx— Tracker Mask Decoder (video tracking)
   *   [4] = text_encoder.onnx   — CLIP Text Encoder (optional)
   */
  std::vector<std::string> model_value_;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

// ==================== SAM 3 Graph ====================

/**
 * @brief SAM 3 Graph — Real Architecture
 *
 * Implements the full SAM 3 decoupled detector-tracker pipeline:
 *
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │  Image ──► Preprocess ──► PerceptionEncoder (Shared ViT) ─────┐│
 *   │                     ┌───────────────────────┴──────┐           ││
 *   │                     ▼                              ▼           ││
 *   │  ┌──────────────────────────┐   ┌──────────────────────────┐   ││
 *   │  │     Detector Path        │   │     Tracker Path          │   ││
 *   │  │  (Image mode)            │   │  (Video mode)             │   ││
 *   │  │                          │   │                          │   ││
 *   │  │  Text ──► TextEncoder ──┐│   │  MemoryEncoder ──► Memory│   ││
 *   │  │  Exemplar──► ExemplarEnc││   │  Bank ──► TrackMaskDecode│   ││
 *   │  │                         ▼│   │           │              │   ││
 *   │  │  DetectorDecoder (DETR)  │   │           ▼              │   ││
 *   │  │       │                  │   │  ┌──────────────┐        │   ││
 *   │  │  PresenceHead ──► Concept│   │  │PostProcess   │        │   ││
 *   │  │  Matcher ──► PostProcess │   │  │(track ID vis)│        │   ││
 *   │  └──────────────────────────┘   └──────────────────────────┘   ││
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * ─── ONNX Model Files (Community Standard) ─────────────────────────
 *
 * SAM 3 follows a 3-model ONNX convention used by wkentaro/sam3-onnx,
 * samexporter, and other community projects. Our internal DAG decomposes
 * these into finer-grained nodes for pipeline flexibility:
 *
 *   ┌──────────────────────┬──────────────────────────────────────────┐
 *   │ ONNX File            │ Used by Our C++ Nodes                    │
 *   ├──────────────────────┼──────────────────────────────────────────┤
 *   │ sam3_image_encoder   │ Sam3PerceptionEncoder (shared ViT       │
 *   │ .onnx                │ backbone, multi-scale FPN outputs)      │
 *   ├──────────────────────┼──────────────────────────────────────────┤
 *   │ sam3_language_       │ Sam3LanguageEncodeNode (CLIP-style      │
 *   │ encoder.onnx         │ text → embeddings)                      │
 *   ├──────────────────────┼──────────────────────────────────────────┤
 *   │ sam3_decoder.onnx    │ Sam3DetectorDecoder +                   │
 *   │                      │ Sam3PresenceHead (monolithic decoder    │
 *   │                      │ wraps FusionEncoder + TransformerDec    │
 *   │                      │ + PresenceHead + MaskHead)              │
 *   ├──────────────────────┼──────────────────────────────────────────┤
 *   │ memory_encoder.onnx  │ Sam3MemoryEncoder (video tracking)      │
 *   ├──────────────────────┼──────────────────────────────────────────┤
 *   │ tracker_decoder.onnx │ Sam3TrackerMaskDecoder (video tracking) │
 *   └──────────────────────┴──────────────────────────────────────────┘
 *
 * Pre-exported models (can skip export step):
 *   - wkentaro/sam3-onnx-models-v0.3.0       (HuggingFace)
 *   - vietanhdev/segment-anything-3-onnx     (HuggingFace)
 *
 * See tools/export_sam3_onnx/ for the export script.
 *
 * model_value[] mapping (setInferParam):
 *   [0] = sam3_image_encoder.onnx
 *   [1] = sam3_decoder.onnx
 *   [2] = memory_encoder.onnx         (optional, video)
 *   [3] = tracker_decoder.onnx        (optional, video)
 *   [4] = sam3_language_encoder.onnx  (optional, text prompts)
 */
class NNDEPLOY_CC_API SAM3Graph : public dag::Graph {
 public:
  SAM3Graph(const std::string& name) : dag::Graph(name) {
    key_ = "nndeploy::segment::SAM3Graph";
    desc_ = "SAM 3 Graph: decoupled DETR detector + SAM 2 tracker, shared PE.";
    param_ = std::make_shared<Sam3GraphParam>();
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<Sam3ConceptParam>();
    this->setOutputTypeInfo<cv::Mat>();
    initDynamicsGraphNodes();
    defaultParam();
  }

  SAM3Graph(const std::string& name, std::vector<dag::Edge*> inputs,
            std::vector<dag::Edge*> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::segment::SAM3Graph";
    desc_ = "SAM 3 Graph: decoupled DETR detector + SAM 2 tracker, shared PE.";
    param_ = std::make_shared<Sam3GraphParam>();
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<Sam3ConceptParam>();
    this->setOutputTypeInfo<cv::Mat>();
    initDynamicsGraphNodes();
    defaultParam();
  }

  // Set model paths for all five ONNX models
  base::Status setInferParam(base::InferenceType inference_type,
                             base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::vector<std::string>& model_value);

  base::Status defaultParam() override;

  // Override deserialize to support model path configuration from JSON
  // After the base class deserializes param_, this reads model_value_ from
  // Sam3GraphParam and calls setInferParam() to configure the internal
  // infer nodes. This enables full JSON workflow configuration.
  using dag::Graph::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;

  // Set text concepts for detection
  base::Status setConcepts(const std::vector<std::string>& concepts) {
    if (concept_encode_node_ != nullptr) {
      return concept_encode_node_->setConcepts(concepts);
    }
    return base::kStatusCodeErrorInvalidValue;
  }

  // ==================== Forward Methods ====================

  /**
   * @brief Single image forward pass (Image mode)
   *
   * Full pipeline: PE → DetectorDecoder → PresenceHead → ConceptMatcher →
   * PostProcess
   */
  std::vector<dag::Edge*> forward(std::vector<dag::Edge*> inputs) override;

  /**
   * @brief Single image with exemplar prompts
   */
  std::vector<dag::Edge*> forwardWithExemplars(
      cv::Mat& image, std::vector<cv::Mat>& exemplar_images);

  /**
   * @brief Video forward with text concept prompts
   *
   * For frame 0: Detector path (PE → Detector → PresenceHead → ConceptMatcher)
   * For frames 1..N: Tracker path (PE → MemoryEncoder → MemoryManager →
   * TrackerMaskDecoder)
   *
   * Concept embeddings are cached after first frame for efficiency.
   */
  std::vector<std::vector<dag::Edge*>> forwardVideoWithTextPrompt(
      std::vector<cv::Mat>& video_frames, std::vector<std::string>& concepts,
      std::vector<std::vector<float>> prompts = {});

  /**
   * @brief Video forward with exemplar prompts
   */
  std::vector<std::vector<dag::Edge*>> forwardVideoWithExemplars(
      std::vector<cv::Mat>& video_frames,
      std::vector<cv::Mat>& exemplar_images);

  // Reset video tracking state (call between video clips)
  void resetVideoTracking() {
    active_instances_.clear();
    concept_embeddings_cached_.reset();
    concept_embeddings_cached_valid_ = false;
    next_track_id_ = 0;
    active_frame_count_ = 0;
    if (prev_frame_masks_ != nullptr) {
      delete prev_frame_masks_;
      prev_frame_masks_ = nullptr;
    }
    if (memory_manager_node_ != nullptr) {
      memory_manager_node_->reset();
    }
  }

 private:
  dag::Edge* fusePrompts(dag::Edge* text_emb, dag::Edge* exemplar_emb);

  base::Status initDynamicsGraphNodes();

  // Internal video frame processing
  base::Status processVideoFrame(cv::Mat& frame,
                                 device::Tensor* cached_concept_emb,
                                 int num_concepts, dag::Edge* output_edge,
                                 bool use_tracker = false);

  // Track ID management
  int assignTrackId() { return next_track_id_++; }

  // ==================== Internal Nodes ====================

  // Preprocessing
  dag::Node* preprocess_image_node_ = nullptr;

  // Legacy nodes (backward compatible)
  Sam3ConceptEncodeNode* concept_encode_node_ = nullptr;
  Sam3LanguageEncodeNode* language_encode_node_ = nullptr;
  Sam3ExemplarEncodeNode* exemplar_encode_node_ = nullptr;
  Sam3PostProcess* postprocess_node_ = nullptr;

  // NEW: Real SAM 3 architecture nodes
  Sam3PerceptionEncoder* perception_encoder_node_ = nullptr;  // Shared PE
  Sam3DetectorDecoder* detector_decoder_node_ = nullptr;      // DETR detector
  Sam3PresenceHead* presence_head_node_ = nullptr;            // Presence Head
  Sam3ConceptMatcher* concept_matcher_node_ = nullptr;      // Concept matching
  Sam3MemoryEncoder* memory_encoder_node_ = nullptr;        // Memory encoding
  Sam3TrackerMaskDecoder* tracker_decoder_node_ = nullptr;  // Tracker decoder
  Sam3MemoryManager* memory_manager_node_ = nullptr;        // Memory state mgmt

  // Inference params
  inference::InferenceParam pe_infer_param_;         // model_value[0]
  inference::InferenceParam detector_infer_param_;   // model_value[1]
  inference::InferenceParam memory_encoder_param_;   // model_value[2]
  inference::InferenceParam tracker_decoder_param_;  // model_value[3]

  // Video tracking state
  std::vector<DetectionInstance> active_instances_;
  std::shared_ptr<device::Tensor> concept_embeddings_cached_;
  bool concept_embeddings_cached_valid_ = false;
  int next_track_id_ = 0;
  int active_frame_count_ = 0;
  cv::Mat* prev_frame_masks_ = nullptr;
};

}  // namespace segment
}  // namespace nndeploy

#endif /* _NNDEPLOY_SEGMENT_SAM3_H_ */
