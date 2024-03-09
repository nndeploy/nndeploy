
#include "nndeploy/codec/opencv/opencv_codec.h"

#include "nndeploy/base/file.h"

namespace nndeploy {
namespace codec {

base::Status OpenCvImageDecodeNode::init() {
  if (base::exists(path_)) {
    size_ = 1;
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("path[%s] is not exists!\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
}
base::Status OpenCvImageDecodeNode::deinit() { return base::kStatusCodeOk; }

base::Status OpenCvImageDecodeNode::run() {
  cv::Mat *mat = new cv::Mat(cv::imread(path_));
  width_ = mat->cols;
  height_ = mat->rows;
  outputs_[0]->set(mat, 0, false);
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesDecodeNode::init() {
  index_ = 0;
  if (base::isDirectory(path_)) {
    base::Status status = base::kStatusCodeOk;
    std::vector<std::string> jpg_result;
    base::glob(path_, "*.jpg", jpg_result);
    images_.insert(images_.end(), jpg_result.begin(), jpg_result.end());

    std::vector<std::string> png_result;
    base::glob(path_, "*.png", png_result);
    images_.insert(images_.end(), png_result.begin(), png_result.end());

    std::vector<std::string> jpeg_result;
    base::glob(path_, "*.jpeg", jpeg_result);
    images_.insert(images_.end(), jpeg_result.begin(), jpeg_result.end());

    std::vector<std::string> bmp_result;
    base::glob(path_, ".bmp", bmp_result);
    images_.insert(images_.end(), bmp_result.begin(), bmp_result.end());

    size_ = (int)images_.size();
    if (size_ == 0) {
      NNDEPLOY_LOGE("path[%s] not exist pic!\n", path_.c_str());
      status = base::kStatusCodeErrorInvalidParam;
    }
    return status;
  } else {
    NNDEPLOY_LOGE("path[%s] is not Directory!\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
}
base::Status OpenCvImagesDecodeNode::deinit() {
  images_.clear();
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesDecodeNode::run() {
  if (index_ < size_) {
    std::string image_path = images_[index_];
    cv::Mat *mat = new cv::Mat(cv::imread(image_path));
    outputs_[0]->set(mat, index_, false);
    index_++;
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("Invalid parameter error occurred. index[%d] >=size_[%d].\n ",
                  index_, size_);
    return base::kStatusCodeErrorInvalidParam;
  }
}

base::Status OpenCvVedioDecodeNode::init() {
  index_ = 0;
  if (base::exists(path_)) {
    base::Status status = base::kStatusCodeOk;
    cap_ = new cv::VideoCapture(path_);

    if (!cap_->isOpened()) {
      NNDEPLOY_LOGE("Error: Failed to open video file.\n");
      delete cap_;
      cap_ = nullptr;
      return base::kStatusCodeErrorInvalidParam;
    }

    size_ = (int)cap_->get(cv::CAP_PROP_FRAME_COUNT);
    fps_ = cap_->get(cv::CAP_PROP_FPS);
    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
    NNDEPLOY_LOGI("Video frame count: %d.\n", size_);
    NNDEPLOY_LOGI("Video FPS: %f.\n", fps_);
    NNDEPLOY_LOGI("Video width_: %d.\n", width_);
    NNDEPLOY_LOGI("Video height_: %d.\n", height_);

    return status;
  } else {
    NNDEPLOY_LOGE("path[%s] is not Directory!\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
}
base::Status OpenCvVedioDecodeNode::deinit() {
  if (cap_ != nullptr) {
    delete cap_;
    cap_ = nullptr;
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvVedioDecodeNode::run() {
  if (index_ < size_) {
    cv::Mat *mat = new cv::Mat();
    cap_->read(*mat);
    outputs_[0]->set(mat, index_, false);
    index_++;
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("Invalid parameter error occurred. index[%d] >=size_[%d].\n ",
                  index_, size_);
    return base::kStatusCodeErrorInvalidParam;
  }
}

base::Status OpenCvCameraDecodeNode::init() {
  index_ = 0;
  base::Status status = base::kStatusCodeOk;
  if (path_.empty()) {
    cap_ = new cv::VideoCapture(0);
  } else if (base::isNumeric(path_)) {
    int index = std::stoi(path_);
    cap_ = new cv::VideoCapture(index);
  } else {
    cap_ = new cv::VideoCapture(path_);
  }

  if (!cap_->isOpened()) {
    NNDEPLOY_LOGE("Error: Failed to open video file.\n");
    delete cap_;
    cap_ = nullptr;
    return base::kStatusCodeErrorInvalidParam;
  }

  fps_ = cap_->get(cv::CAP_PROP_FPS);
  size_ = 60 * (int)fps_;
  width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
  height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
  NNDEPLOY_LOGI("Video frame count: %d.\n", size_);
  NNDEPLOY_LOGI("Video FPS: %f.\n", fps_);
  NNDEPLOY_LOGI("Video width_: %d.\n", width_);
  NNDEPLOY_LOGI("Video height_: %d.\n", height_);

  return status;
}
base::Status OpenCvCameraDecodeNode::deinit() {
  if (cap_ != nullptr) {
    delete cap_;
    cap_ = nullptr;
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvCameraDecodeNode::run() {
  if (index_ < size_) {
    cv::Mat *mat = new cv::Mat();
    cap_->read(*mat);
    outputs_[0]->set(mat, index_, false);
    index_++;
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("Invalid parameter error occurred. index[%d] >=size_[%d].\n ",
                  index_, size_);
    return base::kStatusCodeErrorInvalidParam;
  }
}

base::Status OpenCvImageEncodeNode::init() { return base::kStatusCodeOk; }
base::Status OpenCvImageEncodeNode::deinit() { return base::kStatusCodeOk; }

base::Status OpenCvImageEncodeNode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  cv::imwrite(path_, *mat);
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesEncodeNode::init() {
  index_ = 0;
  if (base::isDirectory(path_)) {
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("path[%s] is not Directory!\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
}
base::Status OpenCvImagesEncodeNode::deinit() { return base::kStatusCodeOk; }

base::Status OpenCvImagesEncodeNode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  std::string name = std::to_string(index_);
  std::string full_path = base::joinPath(path_, name);
  cv::imwrite(full_path, *mat);
  index_++;
  return base::kStatusCodeOk;
}

base::Status OpenCvVedioEncodeNode::init() {
  base::Status status = base::kStatusCodeOk;
  if (base::exists(ref_path_)) {
    base::Status status = base::kStatusCodeOk;
    cap_ = new cv::VideoCapture(ref_path_);

    if (!cap_->isOpened()) {
      NNDEPLOY_LOGE("Error: Failed to open video file.\n");
      delete cap_;
      cap_ = nullptr;
      return base::kStatusCodeErrorInvalidParam;
    }

    fps_ = cap_->get(cv::CAP_PROP_FPS);
    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
    NNDEPLOY_LOGI("Video FPS: %f.\n", fps_);
    NNDEPLOY_LOGI("Video width_: %d.\n", width_);
    NNDEPLOY_LOGI("Video height_: %d.\n", height_);

    return status;
  }

  int fourcc =
      cv::VideoWriter::fourcc(fourcc_[0], fourcc_[1], fourcc_[2], fourcc_[3]);
  cv::Size frame_size(width_, height_);
  writer_ = new cv::VideoWriter(path_, fourcc, fps_, frame_size);
  // 检查视频写入对象是否成功打开
  if (!writer_->isOpened()) {
    NNDEPLOY_LOGE("Error: Failed to open output video file.\n");
    delete writer_;
    writer_ = nullptr;
    return base::kStatusCodeErrorInvalidParam;
  }

  return status;
}
base::Status OpenCvVedioEncodeNode::deinit() {
  base::Status status = base::kStatusCodeOk;
  if (!cap_) {
    delete cap_;
    cap_ = nullptr;
  }
  if (!writer_) {
    delete writer_;
    writer_ = nullptr;
  }
  return status;
}

base::Status OpenCvVedioEncodeNode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  writer_->write(*mat);
  return base::kStatusCodeOk;
}

base::Status OpenCvCameraEncodeNode::init() {
  base::Status status = base::kStatusCodeOk;
  return status;
}
base::Status OpenCvCameraEncodeNode::deinit() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status OpenCvCameraEncodeNode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  if (mat != nullptr) {
    cv::imshow(path_, *mat);
  }
  return base::kStatusCodeOk;
}

TypeCreatelDecodeNodeRegister g_type_create_decode_node_register(
    base::kCodecTypeOpenCV, createOpenCvDecodeNode);

DecodeNode *createOpenCvDecodeNode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *output) {
  DecodeNode *temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = new OpenCvImageDecodeNode(flag, name, output);
  } else if (flag == base::kCodecFlagImages) {
    temp = new OpenCvImagesDecodeNode(flag, name, output);
  } else if (flag == base::kCodecFlagVideo) {
    temp = new OpenCvVedioDecodeNode(flag, name, output);
  } else if (flag == base::kCodecFlagCamera) {
    temp = new OpenCvCameraDecodeNode(flag, name, output);
  }

  return temp;
}

TypeCreatelEncodeNodeRegister g_type_create_encode_node_register(
    base::kCodecTypeOpenCV, createOpenCvEncodeNode);

EncodeNode *createOpenCvEncodeNode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *input) {
  EncodeNode *temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = new OpenCvImageEncodeNode(flag, name, input);
  } else if (flag == base::kCodecFlagImages) {
    temp = new OpenCvImagesEncodeNode(flag, name, input);
  } else if (flag == base::kCodecFlagVideo) {
    temp = new OpenCvVedioEncodeNode(flag, name, input);
  } else if (flag == base::kCodecFlagCamera) {
    temp = new OpenCvCameraEncodeNode(flag, name, input);
  }

  return temp;
}

}  // namespace codec
}  // namespace nndeploy