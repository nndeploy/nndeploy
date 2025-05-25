#include "nndeploy/codec/opencv/opencv_codec.h"

#include "nndeploy/base/file.h"

namespace nndeploy {
namespace codec {

base::Status OpenCvImageDecodeNode::init() { return base::kStatusCodeOk; }
base::Status OpenCvImageDecodeNode::deinit() { return base::kStatusCodeOk; }

base::Status OpenCvImageDecodeNode::setPath(const std::string &path) {
  path_ = path;
  if (!base::exists(path_)) {
    NNDEPLOY_LOGE("path[%s] is not exists!\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  path_changed_ = true;
  size_ = 1;
  return base::kStatusCodeOk;
}

base::Status OpenCvImageDecodeNode::run() {
  while (path_.empty() && parallel_type_ == base::kParallelTypePipeline) {
    // NNDEPLOY_LOGE("path[%s] is empty!\n", path_.c_str());
    ;
  }
  cv::Mat *mat = new cv::Mat(cv::imread(path_));
  width_ = mat->cols;
  height_ = mat->rows;
  // NNDEPLOY_LOGE("OpenCvImageDecodeNode::run() width_[%d] height_[%d]\n",
  // width_, height_);
  outputs_[0]->set(mat, false);
  index_++;
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesDecodeNode::init() { return base::kStatusCodeOk; }
base::Status OpenCvImagesDecodeNode::deinit() {
  images_.clear();
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesDecodeNode::setPath(const std::string &path) {
  if (path_ == path) {
    return base::kStatusCodeOk;
  }
  path_ = path;
  index_ = 0;
  images_.clear();
  path_changed_ = true;
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
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesDecodeNode::run() {
  while (path_.empty() && parallel_type_ == base::kParallelTypePipeline) {
    // NNDEPLOY_LOGE("path[%s] is empty!\n", path_.c_str());
    ;
  }
  if (index_ < size_) {
    std::string image_path = images_[index_];
    cv::Mat *mat = new cv::Mat(cv::imread(image_path));
    outputs_[0]->set(mat, false);
    index_++;
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("Invalid parameter error occurred. index[%d] >=size_[%d].\n ",
                  index_, size_);
    return base::kStatusCodeErrorInvalidParam;
  }
}

base::Status OpenCvVedioDecodeNode::init() {
  // index_ = 0;
  // if (base::exists(path_)) {
  //   base::Status status = base::kStatusCodeOk;
  //   cap_ = new cv::VideoCapture(path_);

  //   if (!cap_->isOpened()) {
  //     NNDEPLOY_LOGE("Error: Failed to open video file.\n");
  //     delete cap_;
  //     cap_ = nullptr;
  //     return base::kStatusCodeErrorInvalidParam;
  //   }

  //   size_ = (int)cap_->get(cv::CAP_PROP_FRAME_COUNT);
  //   fps_ = cap_->get(cv::CAP_PROP_FPS);
  //   width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
  //   height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
  //   NNDEPLOY_LOGI("Video frame count: %d.\n", size_);
  //   NNDEPLOY_LOGI("Video FPS: %f.\n", fps_);
  //   NNDEPLOY_LOGI("Video width_: %d.\n", width_);
  //   NNDEPLOY_LOGI("Video height_: %d.\n", height_);

  //   return status;
  // } else {
  //   NNDEPLOY_LOGE("path[%s] is not Directory!\n", path_.c_str());
  //   return base::kStatusCodeErrorInvalidParam;
  // }
  return base::kStatusCodeOk;
}
base::Status OpenCvVedioDecodeNode::deinit() {
  if (cap_ != nullptr) {
    cap_->release();
    delete cap_;
    cap_ = nullptr;
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvVedioDecodeNode::setPath(const std::string &path) {
  NNDEPLOY_LOGI("OpenCvVedioDecodeNode::setPath() path_[%s]\n", path.c_str());
  if (path_ == path) {
    return base::kStatusCodeOk;
  }
  path_ = path;
  if (!base::exists(path_)) {
    NNDEPLOY_LOGE("path[%s] is not exists!\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  index_ = 0;
  if (cap_ != nullptr) {
    cap_->release();
    delete cap_;
    cap_ = nullptr;
  }
  path_changed_ = true;
  cap_ = new cv::VideoCapture();
  if (!cap_->open(path_)) {
    NNDEPLOY_LOGE("can not open video file %s\n", path_.c_str());
    delete cap_;
    cap_ = nullptr;
    return base::kStatusCodeErrorInvalidParam;
  }

  size_ = (int)cap_->get(cv::CAP_PROP_FRAME_COUNT);
  fps_ = cap_->get(cv::CAP_PROP_FPS);
  width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
  height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
  NNDEPLOY_LOGE("Video frame count: %d.\n", size_);
  NNDEPLOY_LOGE("Video FPS: %f.\n", fps_);
  NNDEPLOY_LOGE("Video width_: %d.\n", width_);
  NNDEPLOY_LOGE("Video height_: %d.\n", height_);
  return base::kStatusCodeOk;
}

base::Status OpenCvVedioDecodeNode::run() {
  while (path_.empty() && parallel_type_ == base::kParallelTypePipeline) {
    // NNDEPLOY_LOGE("path[%s] is empty!\n", path_.c_str());
    ;
  }
  if (index_ < size_) {
    cv::Mat *mat = new cv::Mat();
    cap_->read(*mat);
    outputs_[0]->set(mat, false);
    std::string name = "input_" + std::to_string(index_) + ".jpg";
    std::string full_path = base::joinPath("./", name);
    cv::imwrite(full_path, *mat);
    index_++;
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("Invalid parameter error occurred. index[%d] >=size_[%d].\n ",
                  index_, size_);
    return base::kStatusCodeErrorInvalidParam;
  }
}

base::Status OpenCvCameraDecodeNode::init() {
  // index_ = 0;
  // base::Status status = base::kStatusCodeOk;
  // NNDEPLOY_LOGE("OpenCvCameraDecodeNode::init() path_[%s]\n", path_.c_str());
  // if (path_.empty()) {
  //   cap_ = new cv::VideoCapture(0);
  // } else if (base::isNumeric(path_)) {
  //   int index = std::stoi(path_);
  //   cap_ = new cv::VideoCapture(index);
  // } else {
  //   cap_ = new cv::VideoCapture(path_);
  // }

  // if (!cap_->isOpened()) {
  //   NNDEPLOY_LOGE("Error: Failed to open video file.\n");
  //   delete cap_;
  //   cap_ = nullptr;
  //   return base::kStatusCodeErrorInvalidParam;
  // }

  // fps_ = cap_->get(cv::CAP_PROP_FPS);
  // size_ = 25 * 10;
  // width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
  // height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
  // NNDEPLOY_LOGI("Video frame count: %d.\n", size_);
  // NNDEPLOY_LOGI("Video FPS: %f.\n", fps_);
  // NNDEPLOY_LOGI("Video width_: %d.\n", width_);
  // NNDEPLOY_LOGI("Video height_: %d.\n", height_);
  size_ = INT_MAX;
  return base::kStatusCodeOk;
}
base::Status OpenCvCameraDecodeNode::deinit() {
  if (cap_ != nullptr) {
    cap_->release();
    delete cap_;
    cap_ = nullptr;
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvCameraDecodeNode::setPath(const std::string &path) {
  if (path_ == path) {
    return base::kStatusCodeOk;
  }
  path_ = path;
  index_ = 0;
  if (cap_ != nullptr) {
    cap_->release();
    delete cap_;
    cap_ = nullptr;
  }
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
  // size_ = 25 * 10;
  width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
  height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
  NNDEPLOY_LOGI("Video frame count: %d.\n", size_);
  NNDEPLOY_LOGI("Video FPS: %f.\n", fps_);
  NNDEPLOY_LOGI("Video width_: %d.\n", width_);
  NNDEPLOY_LOGI("Video height_: %d.\n", height_);
  // size_ = INT_MAX;
  return base::kStatusCodeOk;
}

base::Status OpenCvCameraDecodeNode::run() {
  base::Status status = base::kStatusCodeOk;
  if (cap_ == nullptr) {
    NNDEPLOY_LOGE("OpenCvCameraDecodeNode::init() path_[%s]\n", path_.c_str());
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
    // size_ = 25 * 10;
    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
    NNDEPLOY_LOGI("Video frame count: %d.\n", size_);
    NNDEPLOY_LOGI("Video FPS: %f.\n", fps_);
    NNDEPLOY_LOGI("Video width_: %d.\n", width_);
    NNDEPLOY_LOGI("Video height_: %d.\n", height_);
  }

  if (index_ < size_) {
    cv::Mat *mat = new cv::Mat();
    cap_->read(*mat);
    outputs_[0]->set(mat, false);
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

base::Status OpenCvImageEncodeNode::setRefPath(const std::string &path) {
  if (ref_path_ == path) {
    return base::kStatusCodeOk;
  }
  ref_path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvImageEncodeNode::setPath(const std::string &path) {
  if (path_ == path) {
    return base::kStatusCodeOk;
  }
  path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvImageEncodeNode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  cv::imwrite(path_, *mat);
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesEncodeNode::init() {
  // index_ = 0;
  // if (base::isDirectory(path_)) {
  //   return base::kStatusCodeOk;
  // } else {
  //   NNDEPLOY_LOGE("path[%s] is not Directory!\n", path_.c_str());
  //   return base::kStatusCodeErrorInvalidParam;
  // }
  return base::kStatusCodeOk;
}
base::Status OpenCvImagesEncodeNode::deinit() { return base::kStatusCodeOk; }

base::Status OpenCvImagesEncodeNode::setRefPath(const std::string &path) {
  if (ref_path_ == path) {
    return base::kStatusCodeOk;
  }
  ref_path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesEncodeNode::setPath(const std::string &path) {
  if (path_ == path) {
    return base::kStatusCodeOk;
  }
  path_ = path;
  index_ = 0;
  if (base::isDirectory(path_)) {
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("path[%s] is not Directory!\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesEncodeNode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  std::string name = std::to_string(index_) + ".jpg";
  std::string full_path = base::joinPath(path_, name);
  cv::imwrite(full_path, *mat);
  index_++;
  return base::kStatusCodeOk;
}

base::Status OpenCvVedioEncodeNode::init() {
  base::Status status = base::kStatusCodeOk;
  // if (base::exists(ref_path_)) {
  //   base::Status status = base::kStatusCodeOk;
  //   cap_ = new cv::VideoCapture(ref_path_);

  //   if (!cap_->isOpened()) {
  //     NNDEPLOY_LOGE("Error: Failed to open video file.\n");
  //     delete cap_;
  //     cap_ = nullptr;
  //     return base::kStatusCodeErrorInvalidParam;
  //   }

  //   fps_ = cap_->get(cv::CAP_PROP_FPS);
  //   width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
  //   height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
  //   NNDEPLOY_LOGI("Video FPS: %f.\n", fps_);
  //   NNDEPLOY_LOGI("Video width_: %d.\n", width_);
  //   NNDEPLOY_LOGI("Video height_: %d.\n", height_);
  // }

  // int fourcc =
  //     cv::VideoWriter::fourcc(fourcc_[0], fourcc_[1], fourcc_[2],
  //     fourcc_[3]);
  // cv::Size frame_size(width_, height_);
  // writer_ = new cv::VideoWriter(path_, fourcc, fps_, frame_size);
  // // 检查视频写入对象是否成功打开
  // if (!writer_->isOpened()) {
  //   NNDEPLOY_LOGE("Error: Failed to open output video file.\n");
  //   delete writer_;
  //   writer_ = nullptr;
  //   return base::kStatusCodeErrorInvalidParam;
  // }

  return status;
}
base::Status OpenCvVedioEncodeNode::deinit() {
  base::Status status = base::kStatusCodeOk;

  if (writer_) {
    NNDEPLOY_LOGI("writer_ is nullptr\n");
    writer_->release();
    delete writer_;
    writer_ = nullptr;
  }
  if (cap_) {
    cap_->release();
    delete cap_;
    cap_ = nullptr;
  }
  return status;
}

base::Status OpenCvVedioEncodeNode::setRefPath(const std::string &path) {
  if (ref_path_ == path) {
    return base::kStatusCodeOk;
  }
  ref_path_ = path;
  path_changed_ = true;
  if (cap_ != nullptr) {
    cap_->release();
    delete cap_;
    cap_ = nullptr;
  }
  if (base::exists(ref_path_)) {
    base::Status status = base::kStatusCodeOk;
    cap_ = new cv::VideoCapture(ref_path_);

    if (!cap_->isOpened()) {
      NNDEPLOY_LOGE("Error: Failed to open video file %s.\n", ref_path_.c_str());
      delete cap_;
      cap_ = nullptr;
      return base::kStatusCodeErrorInvalidParam;
    }

    // size_ = (int)cap_->get(cv::CAP_PROP_FRAME_COUNT);
    fps_ = cap_->get(cv::CAP_PROP_FPS);
    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
    NNDEPLOY_LOGI("Video FPS: %f.\n", fps_);
    NNDEPLOY_LOGI("Video width_: %d.\n", width_);
    NNDEPLOY_LOGI("Video height_: %d.\n", height_);
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvVedioEncodeNode::setPath(const std::string &path) {
  if (path_ == path) {
    return base::kStatusCodeOk;
  }
  if (writer_ != nullptr) {
    writer_->release();
    delete writer_;
    writer_ = nullptr;
  }
  path_ = path;
  path_changed_ = true;
  int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  cv::Size frame_size(width_, height_);
  writer_ = new cv::VideoWriter(path_, fourcc, fps_, frame_size);
  // 检查视频写入对象是否成功打开
  if (!writer_->isOpened()) {
    NNDEPLOY_LOGE("Error: Failed to open output video file %s.\n", path_.c_str());
    writer_->release();
    delete writer_;
    writer_ = nullptr;
    return base::kStatusCodeErrorInvalidParam;
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvVedioEncodeNode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  writer_->write(*mat);
  std::string name = "output_" + std::to_string(index_) + ".jpg";
  std::string full_path = base::joinPath("./", name);
  cv::imwrite(full_path, *mat);
  index_++;
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

base::Status OpenCvCameraEncodeNode::setRefPath(const std::string &path) {
  if (ref_path_ == path) {
    return base::kStatusCodeOk;
  }
  ref_path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvCameraEncodeNode::setPath(const std::string &path) {
  if (path_ == path) {
    return base::kStatusCodeOk;
  }
  path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvCameraEncodeNode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  if (mat != nullptr) {
#if NNDEPLOY_OS_WINDOWS
    cv::imshow(path_, *mat);
#else
    ;
#endif
  }
  return base::kStatusCodeOk;
}

TypeCreatelDecodeNodeRegister g_type_create_decode_node_register(
    base::kCodecTypeOpenCV, createOpenCvDecodeNode);
TypeCreatelDecodeNodeSharedPtrRegister
    g_type_create_decode_node_shared_ptr_register(
        base::kCodecTypeOpenCV, createOpenCvDecodeNodeSharedPtr);

DecodeNode *createOpenCvDecodeNode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *output) {
  DecodeNode *temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = new OpenCvImageDecodeNode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagImages) {
    temp = new OpenCvImagesDecodeNode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagVideo) {
    temp = new OpenCvVedioDecodeNode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagCamera) {
    temp = new OpenCvCameraDecodeNode(name, {}, {output}, flag);
  }

  return temp;
}

std::shared_ptr<DecodeNode> createOpenCvDecodeNodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *output) {
  std::shared_ptr<DecodeNode> temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = std::shared_ptr<OpenCvImageDecodeNode>(
        new OpenCvImageDecodeNode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagImages) {
    temp = std::shared_ptr<OpenCvImagesDecodeNode>(
        new OpenCvImagesDecodeNode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagVideo) {
    temp = std::shared_ptr<OpenCvVedioDecodeNode>(
        new OpenCvVedioDecodeNode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagCamera) {
    temp = std::shared_ptr<OpenCvCameraDecodeNode>(
        new OpenCvCameraDecodeNode(name, {}, {output}, flag));
  }

  return temp;
}

TypeCreatelEncodeNodeRegister g_type_create_encode_node_register(
    base::kCodecTypeOpenCV, createOpenCvEncodeNode);
TypeCreatelEncodeNodeSharedPtrRegister
    g_type_create_encode_node_shared_ptr_register(
        base::kCodecTypeOpenCV, createOpenCvEncodeNodeSharedPtr);

EncodeNode *createOpenCvEncodeNode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *input) {
  EncodeNode *temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = new OpenCvImageEncodeNode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagImages) {
    temp = new OpenCvImagesEncodeNode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagVideo) {
    temp = new OpenCvVedioEncodeNode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagCamera) {
    temp = new OpenCvCameraEncodeNode(name, {input}, {}, flag);
  }

  return temp;
}

std::shared_ptr<EncodeNode> createOpenCvEncodeNodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *input) {
  std::shared_ptr<EncodeNode> temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = std::shared_ptr<OpenCvImageEncodeNode>(
        new OpenCvImageEncodeNode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagImages) {
    temp = std::shared_ptr<OpenCvImagesEncodeNode>(
        new OpenCvImagesEncodeNode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagVideo) {
    temp = std::shared_ptr<OpenCvVedioEncodeNode>(
        new OpenCvVedioEncodeNode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagCamera) {
    temp = std::shared_ptr<OpenCvCameraEncodeNode>(
        new OpenCvCameraEncodeNode(name, {input}, {}, flag));
  }

  return temp;
}

REGISTER_NODE("nndeploy::codec::OpenCvImageDecodeNode", OpenCvImageDecodeNode);
REGISTER_NODE("nndeploy::codec::OpenCvImagesDecodeNode",
              OpenCvImagesDecodeNode);
REGISTER_NODE("nndeploy::codec::OpenCvVedioDecodeNode", OpenCvVedioDecodeNode);
REGISTER_NODE("nndeploy::codec::OpenCvCameraDecodeNode",
              OpenCvCameraDecodeNode);
REGISTER_NODE("nndeploy::codec::OpenCvImageEncodeNode", OpenCvImageEncodeNode);
REGISTER_NODE("nndeploy::codec::OpenCvImagesEncodeNode",
              OpenCvImagesEncodeNode);
REGISTER_NODE("nndeploy::codec::OpenCvVedioEncodeNode", OpenCvVedioEncodeNode);
REGISTER_NODE("nndeploy::codec::OpenCvCameraEncodeNode",
              OpenCvCameraEncodeNode);

}  // namespace codec
}  // namespace nndeploy