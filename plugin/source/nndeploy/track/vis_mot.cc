#include "nndeploy/track/vis_mot.h"

namespace nndeploy {
namespace track {

// cv::Scalar VisMOT::GetMOTBoxColor(int idx) {
//   idx = idx * 3;
//   cv::Scalar color =
//       cv::Scalar((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255);
//   return color;
// }

// base::Status VisMOT::run() {
//   cv::Mat *img = inputs_[0]->getCvMat(this);
//   MOTResult *results = (MOTResult *)inputs_[1]->getParam(this);

//   float score_threshold = 0.0;

//   cv::Mat *vis_img = new cv::Mat();
//   img->copyTo(*vis_img);

//   int im_h = img->rows;
//   int im_w = img->cols;
//   float text_scale = std::max(1, static_cast<int>(im_w / 1600.));
//   float text_thickness = 2.;
//   float line_thickness = std::max(1, static_cast<int>(im_w / 500.));

//   for (int i = 0; i < results->boxes.size(); ++i) {
//     if (results->scores[i] < score_threshold) {
//       continue;
//     }
//     const int obj_id = results->ids[i];
//     const float score = results->scores[i];
//     cv::Scalar color = GetMOTBoxColor(obj_id);

//     cv::Point pt1 = cv::Point(results->boxes[i][0], results->boxes[i][1]);
//     cv::Point pt2 = cv::Point(results->boxes[i][2], results->boxes[i][3]);
//     cv::Point id_pt =
//         cv::Point(results->boxes[i][0], results->boxes[i][1] + 10);
//     cv::Point score_pt =
//         cv::Point(results->boxes[i][0], results->boxes[i][1] - 10);
//     cv::rectangle(*vis_img, pt1, pt2, color, line_thickness);
//     std::ostringstream idoss;
//     idoss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
//     idoss << obj_id;
//     std::string id_text = idoss.str();

//     cv::putText(*vis_img, id_text, id_pt, cv::FONT_HERSHEY_PLAIN, text_scale,
//                 color, text_thickness);

//     std::ostringstream soss;
//     soss << std::setiosflags(std::ios::fixed) << std::setprecision(2);
//     soss << score;
//     std::string score_text = soss.str();

//     cv::putText(*vis_img, score_text, score_pt, cv::FONT_HERSHEY_PLAIN,
//                 text_scale, color, text_thickness);
//   }

//   outputs_[0]->set(vis_img, false);
//   return base::kStatusCodeOk;
// }

// cv::Scalar VisBoxMot::GetBoxColor(int idx) {
//   idx = idx * 3;
//   return cv::Scalar((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255);
// }

// base::Status VisBoxMot::run() {
//   cv::Mat *img = inputs_[0]->getCvMat(this);
//   BoxMotResult *result = static_cast<BoxMotResult
//   *>(inputs_[1]->getParam(this));

//   if (!img || img->empty() || !result) {
//     return base::kStatusCodeErrorInvalidValue;
//   }

//   cv::Mat *vis_img = new cv::Mat();
//   img->copyTo(*vis_img);

//   int im_w = img->cols;
//   int im_h = img->rows;
//   float text_scale = std::max(1, static_cast<int>(im_w / 1600.));
//   float text_thickness = 2.0f;
//   float line_thickness = std::max(1, static_cast<int>(im_w / 500.));

//   // BBoxResult stores normalized [0,1] coordinates (yolo.cc divides by
//   // model_w/model_h). Scale to pixel space before drawing, matching
//   // DrawBox's pattern in detect/drawbox.h.
//   float w_ratio = static_cast<float>(im_w);
//   float h_ratio = static_cast<float>(im_h);

//   for (const auto &track : result->tracks_) {
//     cv::Scalar color = GetBoxColor(track.id_);

//     if (track.is_obb_) {
//       // obb_[0]=cx, obb_[1]=cy, obb_[2]=w, obb_[3]=h, obb_[4]=angle
//       // All normalized [0,1] except angle
//       float cx = track.obb_[0] * w_ratio;
//       float cy = track.obb_[1] * h_ratio;
//       float rw = track.obb_[2] * w_ratio;
//       float rh = track.obb_[3] * h_ratio;
//       cv::Point2f center(cx, cy);
//       cv::Size2f size(rw, rh);
//       float angle_deg = track.obb_[4] * 180.0f / 3.14159265f;
//       cv::RotatedRect rot_rect(center, size, angle_deg);

//       cv::Point2f vertices[4];
//       rot_rect.points(vertices);
//       for (int j = 0; j < 4; j++) {
//         cv::line(*vis_img, vertices[j], vertices[(j + 1) % 4], color,
//                  line_thickness);
//       }

//       cv::Point id_pt(vertices[0].x, vertices[0].y - 5);
//       std::ostringstream id_oss;
//       id_oss << "ID:" << track.id_;
//       cv::putText(*vis_img, id_oss.str(), id_pt, cv::FONT_HERSHEY_PLAIN,
//                   text_scale, color, text_thickness);

//       cv::Point score_pt(vertices[0].x, vertices[0].y + 15);
//       std::ostringstream score_oss;
//       score_oss << std::setiosflags(std::ios::fixed) << std::setprecision(2)
//                 << track.confidence_;
//       cv::putText(*vis_img, score_oss.str(), score_pt,
//       cv::FONT_HERSHEY_PLAIN,
//                   text_scale, color, text_thickness);
//     } else {
//       // bbox_[0..3] are normalized [0,1] — scale to pixel coords
//       int x1 = static_cast<int>(track.bbox_[0] * w_ratio);
//       int y1 = static_cast<int>(track.bbox_[1] * h_ratio);
//       int x2 = static_cast<int>(track.bbox_[2] * w_ratio);
//       int y2 = static_cast<int>(track.bbox_[3] * h_ratio);
//       cv::Point pt1(x1, y1);
//       cv::Point pt2(x2, y2);
//       cv::rectangle(*vis_img, pt1, pt2, color, line_thickness);

//       cv::Point id_pt(x1, y1 - 10);
//       std::ostringstream id_oss;
//       id_oss << "ID:" << track.id_;
//       cv::putText(*vis_img, id_oss.str(), id_pt, cv::FONT_HERSHEY_PLAIN,
//                   text_scale, color, text_thickness);

//       cv::Point score_pt(x1, y1 + 15);
//       std::ostringstream score_oss;
//       score_oss << std::setiosflags(std::ios::fixed) << std::setprecision(2)
//                 << track.confidence_;
//       cv::putText(*vis_img, score_oss.str(), score_pt,
//       cv::FONT_HERSHEY_PLAIN,
//                   text_scale, color, text_thickness);
//     }
//   }

//   outputs_[0]->set(vis_img, false);
//   return base::kStatusCodeOk;
// }

REGISTER_NODE("nndeploy::track::VisMOT", VisMOT);
#ifdef ENABLE_NNDEPLOY_PLUGIN_TRACK_BOXMOT
REGISTER_NODE("nndeploy::track::VisBoxMot", VisBoxMot);
#endif

}  // namespace track
}  // namespace nndeploy
