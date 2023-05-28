#ifndef _NNTASK_SOURCE_COMMON_PARAMS_H_
#define _NNTASK_SOURCE_COMMON_PARAMS_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/param.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/type.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/inference.h"
#include "nndeploy/source/inference/inference_param.h"

namespace nntask {
namespace common {

/**
 * @brief Classify result structure for all the image classify models
 *
 */
class NNDEPLOY_CC_API ClassifyResult : public nndeploy::base::Param {
 public:
  ClassifyResult() : Param(){};
  ClassifyResult(std::string name) : Param(name){};

  ~ClassifyResult(){};

  /**
   * @brief Classify param for an image
   *
   */
  std::vector<int32_t> label_ids_;
  /**
   * @brief The confidence for each classify param
   *
   */
  std::vector<float> scores_;
};

/**
 * @brief structure, used in DetectionResult for instance segmentation models
 *
 */
class NNDEPLOY_CC_API MaskResult : nndeploy::base::Param {
 public:
  MaskResult() : Param(){};
  MaskResult(std::string name) : Param(name){};

  ~MaskResult(){};
  /**
   * @brief Mask data buffer
   *
   */
  std::vector<uint8_t> data_;
  /**
   * @brief Shape of mask
   *
   */
  std::vector<int64_t> shape_;  // (H,W) ...
};

/**
 * @brief Detection result structure for all the object detection models and
 * instance segmentation models
 *
 */
class NNDEPLOY_CC_API DetectResult : public nndeploy::base::Param {
 public:
  DetectResult() : Param(){};
  DetectResult(std::string name) : Param(name){};

  ~DetectResult(){};

  /**
   * @brief All the detected object boxes for an input image, the size of
   * `boxes` is the number of detected objects, and the element of `boxes` is a
   * array of 4 float values, means [xmin, ymin, xmax, ymax]
   *
   */
  std::vector<std::array<float, 4>> boxes_;
  /**
   * @brief All the detected rotated object boxes for an input image, the size
   * of `boxes` is the number of detected objects, and the element of
   * `rotated_boxes` is an array of 8 float values, means [x1, y1, x2, y2, x3,
   * y3, x4, y4]
   */
  std::vector<std::array<float, 8>> rotated_boxes_;
  /**
   * @brief The confidence for all the detected objects
   */
  std::vector<float> scores_;
  /**
   * @brief The classify label for all the detected objects
   *
   */
  std::vector<int32_t> label_ids_;
  /**
   * @brief For instance segmentation model, `masks` is the predict mask for
   * all the deteced objects
   */
  std::vector<MaskResult> masks_;
  /**
   * @brief Shows if the DetectionResult has mask
   *
   */
  bool contain_masks_ = false;
};

/**
 * @brief Detection param structure for all the object detection models and
 * instance segmentation models
 */
class NNDEPLOY_CC_API PerceptionResult : public nndeploy::base::Param {
 public:
  PerceptionResult() : Param(){};
  PerceptionResult(std::string name) : Param(name){};

  ~PerceptionResult(){};

  std::vector<float> scores_;

  std::vector<int32_t> label_ids_;
  /**
   * @brief xmin, ymin, xmax, ymax, h, w, l
   *
   */
  std::vector<std::array<float, 7>> boxes_;
  /**
   * @brief cx, cy, cz
   *
   */
  std::vector<std::array<float, 3>> center_;

  std::vector<float> observation_angle_;

  std::vector<float> yaw_angle_;
  /**
   * @brief vx, vy, vz
   *
   */
  std::vector<std::array<float, 3>> velocity_;
};

/**
 *@brief KeyPoint Detection param structure for all the keypoint detection
 * models
 */
class NNDEPLOY_CC_API KeyPointDetectionResult : public nndeploy::base::Param {
 public:
  KeyPointDetectionResult() : Param(){};
  KeyPointDetectionResult(std::string name) : Param(name){};

  ~KeyPointDetectionResult(){};
  /**
   * @brief All the coordinates of detected keypoints for an input image, the
   * size of `keypoints` is num_detected_objects * num_joints, and the element
   * of `keypoint` is a array of 2 float values, means [x, y]
   */
  std::vector<nndeploy::base::Point2f> keypoints_;
  /**
   * @brief The confidence for all the detected points
   *
   */
  std::vector<float> scores_;
  /**
   * @brief Number of joints for a detected object
   *
   */
  int num_joints_ = -1;
};

class NNDEPLOY_CC_API OCRResult : public nndeploy::base::Param {
 public:
  OCRResult() : Param(){};
  OCRResult(std::string name) : Param(name){};

  ~OCRResult(){};
  std::vector<std::array<int, 8>> boxes_;

  std::vector<std::string> text_;
  std::vector<float> rec_scores_;

  std::vector<float> cls_scores_;
  std::vector<int32_t> cls_labels_;

  std::vector<std::array<int, 8>> table_boxes_;
  std::vector<std::string> table_structure_;
  std::string table_html_;
};

/**
 * @brief MOT(Multi-Object Tracking) param structure for all the MOT models
 */
class NNDEPLOY_CC_API MOTResult : public nndeploy::base::Param {
 public:
  MOTResult() : Param(){};
  MOTResult(std::string name) : Param(name){};

  ~MOTResult(){};
  /**
   * @brief All the tracking object boxes for an input image, the size of
   * `boxes` is the number of tracking objects, and the element of `boxes` is a
   * array of 4 float values, means [xmin, ymin, xmax, ymax]
   */
  std::vector<std::array<int, 4>> boxes_;
  /**
   * @brief All the tracking object ids
   */
  std::vector<int> ids_;
  /**
   * @brief The confidence for all the tracking objects
   */
  std::vector<float> scores_;
  /**
   * @brief The classify label id for all the tracking object
   */
  std::vector<int> class_ids_;
};

/**
 * @brief Face detection param structure for all the face detection models
 */
class NNDEPLOY_CC_API FaceDetectionResult : public nndeploy::base::Param {
 public:
  FaceDetectionResult() : Param(){};
  FaceDetectionResult(std::string name) : Param(name){};

  ~FaceDetectionResult(){};
  /**
   * @brief All the detected object boxes for an input image, the size of
   * `boxes` is the number of detected objects, and the element of `boxes` is a
   * array of 4 float values, means [xmin, ymin, xmax, ymax]
   */
  std::vector<std::array<float, 4>> boxes_;
  /**
   * @brief
   * If the model detect face with landmarks, every detected object box
   * correspoing to a landmark, which is a array of 2 float values, means
   * location [x,y]
   */
  std::vector<nndeploy::base::Point2f> landmarks_;
  /**
   * @brief
   * Indicates the confidence of all targets detected from a single image, and
   * the number of elements is consistent with boxes.size()
   */
  std::vector<float> scores_;
  /**
   * @brief
   * `landmarks_per_face` indicates the number of face landmarks for each
   * detected face if the model's output contains face landmarks (such as
   * YOLOv5Face, SCRFD, ...)
   */
  int landmarks_per_face_;
};

/**
 * @brief
 * Face Alignment param structure for all the face alignment models
 */
class NNDEPLOY_CC_API FaceAlignmentResult : public nndeploy::base::Param {
 public:
  FaceAlignmentResult() : Param(){};
  FaceAlignmentResult(std::string name) : Param(name){};

  ~FaceAlignmentResult(){};
  /**
   * @brief
   * All the coordinates of detected landmarks for an input image, and
   * the element of `landmarks` is a array of 2 float values, means [x, y]
   */
  std::vector<nndeploy::base::Point2f> landmarks_;
};

/**
 * @brief
 * Segmentation param structure for all the segmentation models
 */
class NNDEPLOY_CC_API SegmentationResult : public nndeploy::base::Param {
 public:
  SegmentationResult() : Param(){};
  SegmentationResult(std::string name) : Param(name){};

  ~SegmentationResult(){};
  /**
   * @brief
   * `label_map` stores the pixel-level category labels for input image. the
   * number of pixels is equal to label_map.size()
   */
  std::vector<uint8_t> label_map_;
  /**
   * @brief
   * `score_map` stores the probability of the predicted label for each pixel of
   * input image.
   */
  std::vector<float> score_map_;
  /**
   * @brief The output shape, means [H, W]
   *
   */
  std::vector<int64_t> shape_;
  /**
   * @brief SegmentationResult whether containing score_map
   *
   */
  bool contain_score_map_ = false;
};

/**
 * @brief Face recognition param structure for all the Face recognition models
 */
class NNDEPLOY_CC_API FaceRecognitionResult : public nndeploy::base::Param {
 public:
  FaceRecognitionResult() : Param(){};
  FaceRecognitionResult(std::string name) : Param(name){};

  ~FaceRecognitionResult(){};
  /**
   * @brief
   * The feature embedding that represents the final extraction of the
   * face recognition model can be used to calculate the feature similarity
   * between faces.
   */
  std::vector<float> embedding_;
};

/**
 * @brief Matting param structure for all the Matting models
 */
class NNDEPLOY_CC_API MattingResult : public nndeploy::base::Param {
 public:
  MattingResult() : Param(){};
  MattingResult(std::string name) : Param(name){};

  ~MattingResult(){};
  /**
   * @brief
   * `alpha` is a one-dimensional vector, which is the predicted alpha
   * transparency value. The range of values is [0., 1.], and the length is hxw.
   * h, w are the height and width of the input image
   */
  std::vector<float> alpha_;  // h x w
  /**
   * @brief
   * If the model can predict foreground, `foreground` save the predicted
   * foreground image, the shape is [hight,width,channel] generally.
   */
  std::vector<float> foreground_;  // h x w x c (c=3 default)
  /**
   * @brief
   * The shape of output param, when contain_foreground == false, shape only
   * contains (h, w), when contain_foreground == true, shape contains (h, w, c),
   * and c is generally 3
   */
  std::vector<int64_t> shape_;
  /**
   * @brief
   * If the model can predict alpha matte and foreground, contain_foreground =
   * true, default false
   */
  bool contain_foreground_ = false;
};

/**
 * @brief HeadPose param structure for all the headpose models
 */
class NNDEPLOY_CC_API HeadPoseResult : public nndeploy::base::Param {
 public:
  HeadPoseResult() : Param(){};
  HeadPoseResult(std::string name) : Param(name){};

  ~HeadPoseResult(){};
  /**
   * @brief EulerAngles for an input image, and the element of `euler_angles`
   * is a vector, contains {yaw, pitch, roll}
   */
  std::vector<float> euler_angles_;
};

}  // namespace common
}  // namespace nntask

#endif /* _NNTASK_SOURCE_COMMON_PARAMS_H_ */
