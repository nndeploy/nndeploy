#ifndef _NNDEPLOY_KEYPOINT_DRAWKEYPOINT_H_
#define _NNDEPLOY_KEYPOINT_DRAWKEYPOINT_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/keypoint/result.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace keypoint {

// COCO 17 keypoint skeleton connections (16 pairs)
static const int kSkeleton[16][2] = {
    {0, 1},    // nose->left_eye
    {0, 2},    // nose->right_eye
    {1, 3},    // left_eye->left_ear
    {2, 4},    // right_eye->right_ear
    {5, 6},    // left_shoulder->right_shoulder
    {5, 7},    // left_shoulder->left_elbow
    {7, 9},    // left_elbow->left_wrist
    {6, 8},    // right_shoulder->right_elbow
    {8, 10},   // right_elbow->right_wrist
    {11, 12},  // left_hip->right_hip
    {5, 11},   // left_shoulder->left_hip
    {6, 12},   // right_shoulder->right_hip
    {11, 13},  // left_hip->left_knee
    {13, 15},  // left_knee->left_ankle
    {12, 14},  // right_hip->right_knee
    {14, 16},  // right_knee->right_ankle
};

// Limb group colors (BGR): face, arm, leg, torso
static const cv::Scalar kLimbColors[4] = {
    cv::Scalar(239, 188, 100),  // face - aqua/cyan
    cv::Scalar(100, 100, 239),  // arm - red/pink
    cv::Scalar(100, 200, 100),  // leg - green
    cv::Scalar(200, 100, 100),  // torso - blue
};

// Keypoint colors (BGR)
static const cv::Scalar kKpColor(0, 215, 255);  // orange

// Which limb group each skeleton edge belongs to
static const int kLimbGroup[16] = {
    0, 0, 0, 0,  // face (edges 0-3)
    3,           // shoulders (edge 4)
    1, 1, 1, 1,  // arms (edges 5-8)
    3, 3, 3,     // torso (edges 9-11)
    2, 2, 2, 2,  // legs (edges 12-15)
};

// DrawKeypoint: 绘制关键点骨架线+关键点圆点（解耦版）
// 输入：cv::Mat（原始图像）+ KeypointResult（关键点集合，无 bbox）
// 如需同时绘制 bbox，请使用 DrawBBox 节点
class NNDEPLOY_CC_API DrawKeypoint : public dag::Node {
 public:
  DrawKeypoint(const std::string& name) : Node(name) {
    key_ = "nndeploy::keypoint::DrawKeypoint";
    desc_ = "Draw keypoint skeleton from KeypointResult[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<KeypointResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  DrawKeypoint(const std::string& name, std::vector<dag::Edge*> inputs,
               std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::keypoint::DrawKeypoint";
    desc_ = "Draw keypoint skeleton from KeypointResult[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<KeypointResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~DrawKeypoint() {}

  virtual base::Status run() {
    cv::Mat* input_mat = inputs_[0]->get<cv::Mat>(this);
    if (input_mat == nullptr) {
      NNDEPLOY_LOGE("input_mat is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    KeypointResult* result = inputs_[1]->get<KeypointResult>(this);
    if (result == nullptr) {
      NNDEPLOY_LOGE("result is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    NNDEPLOY_LOGD("[%s] detect %zu skeletons (img %dx%d)\n",
                  this->getName().c_str(), result->skeletons_.size(),
                  input_mat->cols, input_mat->rows);
    for (size_t i = 0; i < result->skeletons_.size(); ++i) {
      const auto& skel = result->skeletons_[i];
      int visible = 0;
      for (const auto& kp : skel.keypoints_) {
        if (kp.confidence_ >= 0.5f) visible++;
      }
      NNDEPLOY_LOGD("  [%zu] label=%d score=%.4f keypoints=%d visible=%d\n",
                     i, skel.label_id_, skel.score_,
                     (int)skel.keypoints_.size(), visible);
      for (size_t j = 0; j < skel.keypoints_.size(); ++j) {
        const auto& kp = skel.keypoints_[j];
        NNDEPLOY_LOGD("    kp[%zu] (%.4f,%.4f) conf=%.4f%s\n",
                       j, kp.x_, kp.y_, kp.confidence_,
                       kp.confidence_ < 0.5f ? " [low]" : "");
      }
    }

    float img_w = static_cast<float>(input_mat->cols);
    float img_h = static_cast<float>(input_mat->rows);

    cv::Mat* output_mat = new cv::Mat();
    input_mat->copyTo(*output_mat);

    for (const auto& skel : result->skeletons_) {
      int num_kps = static_cast<int>(skel.keypoints_.size());
      // Draw skeleton edges
      for (int i = 0; i < 16; ++i) {
        int idx1 = kSkeleton[i][0];
        int idx2 = kSkeleton[i][1];
        if (idx1 >= num_kps || idx2 >= num_kps) continue;
        const KeypointKeyPoint& kp1 = skel.keypoints_[idx1];
        const KeypointKeyPoint& kp2 = skel.keypoints_[idx2];
        if (kp1.confidence_ < 0.5f || kp2.confidence_ < 0.5f) continue;
        cv::Point p1(static_cast<int>(kp1.x_ * img_w),
                     static_cast<int>(kp1.y_ * img_h));
        cv::Point p2(static_cast<int>(kp2.x_ * img_w),
                     static_cast<int>(kp2.y_ * img_h));
        cv::line(*output_mat, p1, p2, kLimbColors[kLimbGroup[i]], 2);
      }
      // Draw keypoint circles
      for (int i = 0; i < num_kps; ++i) {
        const KeypointKeyPoint& kp = skel.keypoints_[i];
        if (kp.confidence_ < 0.5f) continue;
        cv::Point pt(static_cast<int>(kp.x_ * img_w),
                     static_cast<int>(kp.y_ * img_h));
        cv::circle(*output_mat, pt, 4, kKpColor, -1);
        cv::circle(*output_mat, pt, 4, cv::Scalar(0, 0, 0), 1);
      }
    }
    outputs_[0]->set(output_mat, false);
    return base::kStatusCodeOk;
  }
};

}  // namespace keypoint
}  // namespace nndeploy

#endif /* _NNDEPLOY_KEYPOINT_DRAWKEYPOINT_H_ */
