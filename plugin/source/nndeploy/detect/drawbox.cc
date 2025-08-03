
#include "nndeploy/detect/drawbox.h"

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/device/device.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace detect {

// DrawBox::DrawBox(const std::string &name,
//                          std::initializer_list<dag::Edge *> inputs,
//                          std::initializer_list<dag::Edge *> outputs)
//     : Node(name, inputs, outputs) {}
// DrawBox::~DrawBox() {}

// base::Status DrawBox::run() {
//   cv::Mat *input_mat = inputs_[0]->getCvMat(this);
//   DetectResult *result = (DetectResult *)inputs_[1]->getParam(this);
//   float w_ratio = float(input_mat->cols);
//   float h_ratio = float(input_mat->rows);
//   const int CNUM = 80;
//   cv::RNG rng(0xFFFFFFFF);
//   cv::Scalar_<int> randColor[CNUM];
//   for (int i = 0; i < CNUM; i++)
//     rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
//   int i = -1;
//   for (auto bbox : result->bboxs_) {
//     std::array<float, 4> box;
//     box[0] = bbox.bbox_[0];  // 640.0;
//     box[2] = bbox.bbox_[2];  // 640.0;
//     box[1] = bbox.bbox_[1];  // 640.0;
//     box[3] = bbox.bbox_[3];  // 640.0;
//     box[0] *= w_ratio;
//     box[2] *= w_ratio;
//     box[1] *= h_ratio;
//     box[3] *= h_ratio;
//     int width = box[2] - box[0];
//     int height = box[3] - box[1];
//     int id = bbox.label_id_;
//     // NNDEPLOY_LOGE("box[0]:%f, box[1]:%f, width :%d, height :%d\n", box[0],
//     //               box[1], width, height);
//     cv::Point p = cv::Point(box[0], box[1]);
//     cv::Rect rect = cv::Rect(box[0], box[1], width, height);
//     cv::rectangle(*input_mat, rect, randColor[id], 2);
//     std::string text = " ID:" + std::to_string(id);
//     cv::putText(*input_mat, text, p, cv::FONT_HERSHEY_PLAIN, 1,
//     randColor[id]);
//   }
//   cv::Mat *output_mat = new cv::Mat(*input_mat);
//   outputs_[0]->set(output_mat,  false);
//   return base::kStatusCodeOk;
// }

// YoloMultiConvDrawBox::YoloMultiConvDrawBox(
//     const std::string &name, std::initializer_list<dag::Edge *> inputs,
//     std::initializer_list<dag::Edge *> outputs)
//     : Node(name, inputs, outputs) {}
// YoloMultiConvDrawBox::~YoloMultiConvDrawBox() {}

// base::Status YoloMultiConvDrawBox::run() {
//   cv::Mat *input_mat = inputs_[0]->getCvMat(this);
//   DetectResult *result = (DetectResult *)inputs_[1]->getParam(this);
//   float w_ratio = float(input_mat->cols);
//   float h_ratio = float(input_mat->rows);
//   const int CNUM = 80;
//   cv::RNG rng(0xFFFFFFFF);
//   cv::Scalar_<int> randColor[CNUM];
//   for (int i = 0; i < CNUM; i++)
//     rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
//   int i = -1;
//   for (auto bbox : result->bboxs_) {
//     std::array<float, 4> box;
//     box[0] = bbox.bbox_[0];  // 640.0;
//     box[2] = bbox.bbox_[2];  // 640.0;
//     box[1] = bbox.bbox_[1];  // 640.0;
//     box[3] = bbox.bbox_[3];  // 640.0;
//     int width = box[2] - box[0];
//     int height = box[3] - box[1];
//     int id = bbox.label_id_;
//     NNDEPLOY_LOGE("box[0]:%f, box[1]:%f, width :%d, height :%d\n", box[0],
//                   box[1], width, height);
//     cv::Point p = cv::Point(box[0], box[1]);
//     cv::Rect rect = cv::Rect(box[0], box[1], width, height);
//     cv::rectangle(*input_mat, rect, randColor[id], 2);
//     std::string text = " ID:" + std::to_string(id);
//     cv::putText(*input_mat, text, p, cv::FONT_HERSHEY_PLAIN, 1,
//     randColor[id]);
//   }
//   cv::Mat *output_mat = new cv::Mat(*input_mat);
//   outputs_[0]->set(output_mat,  false);
//   return base::kStatusCodeOk;
// }

// }  // namespace detect
// }  // namespace nndeploy

// class DrawBox : public dag::Node {
//  public:
//   DrawBox(const std::string &name,
//               std::initializer_list<dag::Edge *> inputs,
//               std::initializer_list<dag::Edge *> outputs);
//   virtual ~DrawBox();

//   virtual base::Status run();
// };

// class YoloMultiConvDrawBox : public dag::Node {
//  public:
//   YoloMultiConvDrawBox(const std::string &name,
//                            std::initializer_list<dag::Edge *> inputs,
//                            std::initializer_list<dag::Edge *> outputs);
//   virtual ~YoloMultiConvDrawBox();

//   virtual base::Status run();
// };

REGISTER_NODE("nndeploy::detect::DrawBox", DrawBox);
REGISTER_NODE("nndeploy::detect::YoloMultiConvDrawBox",
              YoloMultiConvDrawBox);

}  // namespace detect
}  // namespace nndeploy
