#include "nndeploy/track/vis_mot.h"

namespace nndeploy {
namespace track {

cv::Scalar VisMOT::GetMOTBoxColor(int idx) {
  idx = idx * 3;
  cv::Scalar color =
      cv::Scalar((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255);
  return color;
}

base::Status VisMOT::run() {
  cv::Mat *img = inputs_[0]->getCvMat(this);
  MOTResult *results = (MOTResult *)inputs_[1]->getParam(this);

  float score_threshold = 0.0;

  cv::Mat *vis_img = new cv::Mat();
  img->copyTo(*vis_img);

  int im_h = img->rows;
  int im_w = img->cols;
  float text_scale = std::max(1, static_cast<int>(im_w / 1600.));
  float text_thickness = 2.;
  float line_thickness = std::max(1, static_cast<int>(im_w / 500.));

  for (int i = 0; i < results->boxes.size(); ++i) {
    if (results->scores[i] < score_threshold) {
      continue;
    }
    const int obj_id = results->ids[i];
    const float score = results->scores[i];
    cv::Scalar color = GetMOTBoxColor(obj_id);

    cv::Point pt1 = cv::Point(results->boxes[i][0], results->boxes[i][1]);
    cv::Point pt2 = cv::Point(results->boxes[i][2], results->boxes[i][3]);
    cv::Point id_pt =
        cv::Point(results->boxes[i][0], results->boxes[i][1] + 10);
    cv::Point score_pt =
        cv::Point(results->boxes[i][0], results->boxes[i][1] - 10);
    cv::rectangle(*vis_img, pt1, pt2, color, line_thickness);
    std::ostringstream idoss;
    idoss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
    idoss << obj_id;
    std::string id_text = idoss.str();

    cv::putText(*vis_img, id_text, id_pt, cv::FONT_HERSHEY_PLAIN, text_scale,
                color, text_thickness);

    std::ostringstream soss;
    soss << std::setiosflags(std::ios::fixed) << std::setprecision(2);
    soss << score;
    std::string score_text = soss.str();

    cv::putText(*vis_img, score_text, score_pt, cv::FONT_HERSHEY_PLAIN,
                text_scale, color, text_thickness);
  }

  outputs_[0]->set(vis_img, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::track::VisMOT", VisMOT);

}  // namespace track
}  // namespace nndeploy