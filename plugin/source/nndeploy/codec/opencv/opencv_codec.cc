#include "nndeploy/codec/opencv/opencv_codec.h"

#include "nndeploy/base/file.h"

namespace nndeploy {
namespace codec {

base::Status OpenCvImageDecode::init() { return base::kStatusCodeOk; }
base::Status OpenCvImageDecode::deinit() { return base::kStatusCodeOk; }

base::Status OpenCvImageDecode::setPath(const std::string &path) {
  if (!base::exists(path)) {
    NNDEPLOY_LOGE("path[%s] is not exists!\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  if (parallel_type_ == base::kParallelTypePipeline) {
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      path_ = path;
      path_changed_ = true;
      path_ready_ = true;  // 设置标志
    }
    path_cv_.notify_one();  // 通知等待的线程
  } else {
    path_ = path;
    path_changed_ = true;
    path_ready_ = true;  // 设置标志
  }
  if (size_ < 1) {
    size_ = 1;
  }
  loop_count_ = size_;
  return base::kStatusCodeOk;
}

base::Status OpenCvImageDecode::run() {
  // while (path_.empty() && parallel_type_ == base::kParallelTypePipeline) {
  //   // NNDEPLOY_LOGE("path[%s] is empty!\n", path_.c_str());
  //   ;
  // }
  // TODO:
  if (index_ == 0 && parallel_type_ == base::kParallelTypePipeline) {
    // NNDEPLOY_LOGI("OpenCvImageDecode::run() path_[%s]\n", path_.c_str());
    std::unique_lock<std::mutex> lock(path_mutex_);
    // 关键：使用lambda检查条件
    path_cv_.wait(lock, [this] { return path_ready_; });
  }
  // NNDEPLOY_LOGI("OpenCvImageDecode::run() path_[%s]\n", path_.c_str());
  cv::Mat *mat = new cv::Mat(cv::imread(path_));
  if (mat == nullptr) {
    NNDEPLOY_LOGE("cv::imread failed! path[%s]\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  width_ = mat->cols;
  height_ = mat->rows;
  // NNDEPLOY_LOGE("OpenCvImageDecode::run() width_[%d] height_[%d]\n",
  // width_, height_);
  outputs_[0]->set(mat, false);
  index_++;
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesDecode::init() { return base::kStatusCodeOk; }
base::Status OpenCvImagesDecode::deinit() {
  images_.clear();
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesDecode::setPath(const std::string &path) {
  if (parallel_type_ == base::kParallelTypePipeline) {
    std::lock_guard<std::mutex> lock(path_mutex_);
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
    } else {
      NNDEPLOY_LOGE("path[%s] is not Directory!\n", path_.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    path_ready_ = true;     // 设置标志
    path_cv_.notify_one();  // 通知等待的线程
  } else {
    path_ = path;
    index_ = 0;
    images_.clear();
    path_changed_ = true;
    path_ready_ = true;  // 设置标志
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
  loop_count_ = size_;
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesDecode::run() {
  // while (path_.empty() && parallel_type_ == base::kParallelTypePipeline) {
  //   // NNDEPLOY_LOGE("path[%s] is empty!\n", path_.c_str());
  //   ;
  // }
  if (index_ == 0 && parallel_type_ == base::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(path_mutex_);
    // 关键：使用lambda检查条件
    path_cv_.wait(lock, [this] { return path_ready_; });
  }
  if (index_ < size_) {
    std::string image_path = images_[index_];
    cv::Mat *mat = new cv::Mat(cv::imread(image_path));
    outputs_[0]->set(mat, false);
    index_++;
  } else {
    NNDEPLOY_LOGI("Invalid parameter error occurred. index[%d] >=size_[%d].\n ",
                  index_, size_);
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvVideoDecode::init() { return base::kStatusCodeOk; }
base::Status OpenCvVideoDecode::deinit() {
  if (cap_ != nullptr) {
    cap_->release();
    delete cap_;
    cap_ = nullptr;
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvVideoDecode::setPath(const std::string &path) {
  if (parallel_type_ == base::kParallelTypePipeline) {
    std::lock_guard<std::mutex> lock(path_mutex_);
    path_ = path;
    
    bool is_url = path_.find("://") != std::string::npos;
    if (!is_url && !base::exists(path_)) {
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
      NNDEPLOY_LOGE("can not open video file/stream %s\n", path_.c_str());
      delete cap_;
      cap_ = nullptr;
      return base::kStatusCodeErrorInvalidParam;
    }
    
    if (is_url) {
      cap_->set(cv::CAP_PROP_BUFFERSIZE, 3);
    }
    size_ = (int)cap_->get(cv::CAP_PROP_FRAME_COUNT);
    fps_ = cap_->get(cv::CAP_PROP_FPS);
    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
    
    if (is_url && size_ <= 0) {
      size_ = INT_MAX;
    }
    
    path_ready_ = true;     // 设置标志
    path_cv_.notify_one();  // 通知等待的线程
  } else {
    path_ = path;
    
    bool is_url = path_.find("://") != std::string::npos;
    if (!is_url && !base::exists(path_)) {
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
      NNDEPLOY_LOGE("can not open video file/stream %s\n", path_.c_str());
      delete cap_;
      cap_ = nullptr;
      return base::kStatusCodeErrorInvalidParam;
    }
    
    if (is_url) {
      cap_->set(cv::CAP_PROP_BUFFERSIZE, 3);
    }
    size_ = (int)cap_->get(cv::CAP_PROP_FRAME_COUNT);
    fps_ = cap_->get(cv::CAP_PROP_FPS);
    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
    
    if (is_url && size_ <= 0) {
      size_ = INT_MAX;
    }
    
    path_ready_ = true;  // 设置标志
  }
  // NNDEPLOY_LOGE("Video frame count: %d.\n", size_);
  // NNDEPLOY_LOGE("Video FPS: %f.\n", fps_);
  // NNDEPLOY_LOGE("Video width_: %d.\n", width_);
  // NNDEPLOY_LOGE("Video height_: %d.\n", height_);
  loop_count_ = size_;
  return base::kStatusCodeOk;
}

base::Status OpenCvVideoDecode::run() {
  // while (path_.empty() && parallel_type_ == base::kParallelTypePipeline) {
  //   // NNDEPLOY_LOGE("path[%s] is empty!\n", path_.c_str());
  //   ;
  // }
  if (index_ == 0 && parallel_type_ == base::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(path_mutex_);
    path_cv_.wait(lock, [this] { return path_ready_; });
  }
  
  if (cap_ == nullptr || !cap_->isOpened()) {
    NNDEPLOY_LOGW("VideoCapture is not opened\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  
  cv::Mat *mat = new cv::Mat();
  bool success = cap_->read(*mat);
  
  if (success && !mat->empty()) {
    outputs_[0]->set(mat, false);
    index_++;
    
    if (size_ != INT_MAX && index_ >= size_) {
      cap_->release();
    }
  } else {
    if (size_ == INT_MAX) {
      NNDEPLOY_LOGW("Network stream read failed, attempting to reconnect...\n");
      delete mat;
      
      cap_->release();
      delete cap_;
      cap_ = new cv::VideoCapture();
      cap_->set(cv::CAP_PROP_BUFFERSIZE, 3);
      cap_->open(path_);
      
      if (!cap_->isOpened()) {
        NNDEPLOY_LOGE("Failed to reconnect to stream %s\n", path_.c_str());
        return base::kStatusCodeErrorInvalidParam;
      }
      
      cv::Mat *retry_mat = new cv::Mat();
      if (cap_->read(*retry_mat) && !retry_mat->empty()) {
        outputs_[0]->set(retry_mat, false);
        index_++;
      } else {
        delete retry_mat;
        return base::kStatusCodeErrorInvalidParam;
      }
    } else {
      delete mat;
      NNDEPLOY_LOGW("Invalid parameter error occurred. index[%d] >=size_[%d].\n ",
                    index_, size_);
    }
  }

  return base::kStatusCodeOk;
}

base::Status OpenCvCameraDecode::init() {
  size_ = 0;
  return base::kStatusCodeOk;
}
base::Status OpenCvCameraDecode::deinit() {
  if (cap_ != nullptr) {
    cap_->release();
    delete cap_;
    cap_ = nullptr;
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvCameraDecode::setPath(const std::string &path) {
  if (parallel_type_ == base::kParallelTypePipeline) {
    std::lock_guard<std::mutex> lock(path_mutex_);
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
    size_ = INT_MAX;
    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
    path_ready_ = true;     // 设置标志
    path_cv_.notify_one();  // 通知等待的线程
  } else {
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
    size_ = INT_MAX;
    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
    path_ready_ = true;  // 设置标志
  }
  // NNDEPLOY_LOGI("Video frame count: %d.\n", size_);
  // NNDEPLOY_LOGI("Video FPS: %f.\n", fps_);
  // NNDEPLOY_LOGI("Video width_: %d.\n", width_);
  // NNDEPLOY_LOGI("Video height_: %d.\n", height_);
  loop_count_ = size_;
  return base::kStatusCodeOk;
}

base::Status OpenCvCameraDecode::run() {
  base::Status status = base::kStatusCodeOk;
  // while (size_ == 0 && parallel_type_ == base::kParallelTypePipeline) {
  //   // NNDEPLOY_LOGE("path[%s] is empty!\n", path_.c_str());
  //   ;
  // }
  if (index_ == 0 && parallel_type_ == base::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(path_mutex_);
    // 关键：使用lambda检查条件
    path_cv_.wait(lock, [this] { return path_ready_; });
  }
  if (index_ < size_) {
    cv::Mat *mat = new cv::Mat();
    cap_->read(*mat);
    outputs_[0]->set(mat, false);
    index_++;
    if (index_ == size_) {
      cap_->release();
    }
  } else {
    NNDEPLOY_LOGW("Invalid parameter error occurred. index[%d] >=size_[%d].\n ",
                  index_, size_);
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvImageEncode::init() { return base::kStatusCodeOk; }
base::Status OpenCvImageEncode::deinit() { return base::kStatusCodeOk; }

base::Status OpenCvImageEncode::setRefPath(const std::string &path) {
  ref_path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvImageEncode::setPath(const std::string &path) {
  path_ = path;
  path_changed_ = true;
  size_ = 1;
  return base::kStatusCodeOk;
}

base::Status OpenCvImageEncode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  cv::imwrite(path_, *mat);
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesEncode::init() { return base::kStatusCodeOk; }
base::Status OpenCvImagesEncode::deinit() { return base::kStatusCodeOk; }

base::Status OpenCvImagesEncode::setRefPath(const std::string &path) {
  ref_path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvImagesEncode::setPath(const std::string &path) {
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

base::Status OpenCvImagesEncode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  std::string name = std::to_string(index_) + ".jpg";
  std::string full_path = base::joinPath(path_, name);
  cv::imwrite(full_path, *mat);
  index_++;
  return base::kStatusCodeOk;
}

base::Status OpenCvVideoEncode::init() {
  base::Status status = base::kStatusCodeOk;
  return status;
}
base::Status OpenCvVideoEncode::deinit() {
  base::Status status = base::kStatusCodeOk;
  if (writer_) {
    if (writer_->isOpened()) {
      NNDEPLOY_LOGI("OpenCvVideoEncode::deinit() writer_->release()\n");
      writer_->release();
    }
    delete writer_;
    writer_ = nullptr;
  }
  return status;
}

base::Status OpenCvVideoEncode::setRefPath(const std::string &path) {
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
      NNDEPLOY_LOGE("Error: Failed to open video file %s.\n",
                    ref_path_.c_str());
      delete cap_;
      cap_ = nullptr;
      return base::kStatusCodeErrorInvalidParam;
    }

    size_ = (int)cap_->get(cv::CAP_PROP_FRAME_COUNT);
    fps_ = cap_->get(cv::CAP_PROP_FPS);
    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);

    cap_->release();
    delete cap_;
    cap_ = nullptr;
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvVideoEncode::setPath(const std::string &path) {
  if (writer_ != nullptr) {
    if (writer_->isOpened()) {
      NNDEPLOY_LOGI("OpenCvVideoEncode::deinit() writer_->release()\n");
      writer_->release();
    }
    delete writer_;
    writer_ = nullptr;
  }
  path_ = path;
  path_changed_ = true;
  if (width_ <= 0 || height_ <= 0) {
    return base::kStatusCodeOk;
  }
  cv::Size frame_size(width_, height_);
  int fourcc =
      cv::VideoWriter::fourcc(fourcc_[0], fourcc_[1], fourcc_[2], fourcc_[3]);
  writer_ = new cv::VideoWriter();
  writer_->open(path_, fourcc, fps_, frame_size, true);
  if (!writer_->isOpened()) {
    NNDEPLOY_LOGE("Error: Failed to open output video file %s.\n",
                  path_.c_str());
    writer_->release();
    delete writer_;
    writer_ = nullptr;
    return base::kStatusCodeErrorInvalidParam;
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvVideoEncode::run() {
  if (writer_ == nullptr && width_ <= 0 && height_ <= 0) {
    cv::Mat *first = inputs_[0]->getCvMat(this);
    if (first == nullptr || first->empty()) {
      NNDEPLOY_LOGE("OpenCvVideoEncode: first frame unavailable for lazy init\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    width_ = first->cols;
    height_ = first->rows;
    cv::Size frame_size(width_, height_);
    int fourcc =
        cv::VideoWriter::fourcc(fourcc_[0], fourcc_[1], fourcc_[2], fourcc_[3]);
    writer_ = new cv::VideoWriter();
    writer_->open(path_, fourcc, fps_, frame_size, true);
    if (!writer_->isOpened()) {
      NNDEPLOY_LOGE("Error: Failed to open output video file %s (lazy init).\n",
                    path_.c_str());
      writer_->release();
      delete writer_;
      writer_ = nullptr;
      return base::kStatusCodeErrorInvalidParam;
    }
  }
  if (size_ == 0 || index_ < size_) {
    cv::Mat *mat = inputs_[0]->getCvMat(this);
    if (mat != nullptr) {
      writer_->write(*mat);
    }
    index_++;
    if (size_ > 0 && index_ == size_) {
      writer_->release();
    }
  } else {
    NNDEPLOY_LOGW("Invalid parameter error occurred. index[%d] >=size_[%d].\n ",
                  index_, size_);
  }

  return base::kStatusCodeOk;
}

base::Status OpenCvCameraEncode::init() {
  base::Status status = base::kStatusCodeOk;
  return status;
}
base::Status OpenCvCameraEncode::deinit() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status OpenCvCameraEncode::setRefPath(const std::string &path) {
  if (ref_path_ == path) {
    return base::kStatusCodeOk;
  }
  ref_path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvCameraEncode::setPath(const std::string &path) {
  if (path_ == path) {
    return base::kStatusCodeOk;
  }
  path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvCameraEncode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  if (mat != nullptr) {
#if NNDEPLOY_OS_WINDOWS || NNDEPLOY_OS_MACOS
    cv::imshow(path_, *mat);
#else
    ;
#endif
    // cv::imshow(path_, *mat);
  }
  return base::kStatusCodeOk;
}

// ==================== OpenCvStreamDecode ====================

base::Status OpenCvStreamDecode::init() {
  size_ = INT_MAX;
  return base::kStatusCodeOk;
}

base::Status OpenCvStreamDecode::deinit() {
  if (cap_ != nullptr) {
    cap_->release();
    delete cap_;
    cap_ = nullptr;
  }
  return base::kStatusCodeOk;
}

base::Status OpenCvStreamDecode::setPath(const std::string &path) {
  if (parallel_type_ == base::kParallelTypePipeline) {
    std::lock_guard<std::mutex> lock(path_mutex_);
    path_ = path;
    index_ = 0;
    if (cap_ != nullptr) {
      cap_->release();
      delete cap_;
      cap_ = nullptr;
    }
    path_changed_ = true;

    cap_ = new cv::VideoCapture();
    cap_->set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (!cap_->open(path_, cv::CAP_FFMPEG)) {
      NNDEPLOY_LOGE("can not open stream %s\n", path_.c_str());
      delete cap_;
      cap_ = nullptr;
      return base::kStatusCodeErrorInvalidParam;
    }
    size_ = INT_MAX;
    fps_ = cap_->get(cv::CAP_PROP_FPS);
    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
    path_ready_ = true;
    path_cv_.notify_one();
  } else {
    path_ = path;
    index_ = 0;
    if (cap_ != nullptr) {
      cap_->release();
      delete cap_;
      cap_ = nullptr;
    }
    path_changed_ = true;

    cap_ = new cv::VideoCapture();
    cap_->set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (!cap_->open(path_, cv::CAP_FFMPEG)) {
      NNDEPLOY_LOGE("can not open stream %s\n", path_.c_str());
      delete cap_;
      cap_ = nullptr;
      return base::kStatusCodeErrorInvalidParam;
    }
    size_ = INT_MAX;
    fps_ = cap_->get(cv::CAP_PROP_FPS);
    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
    path_ready_ = true;
  }
  loop_count_ = size_;
  return base::kStatusCodeOk;
}

base::Status OpenCvStreamDecode::run() {
  if (index_ == 0 && parallel_type_ == base::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(path_mutex_);
    path_cv_.wait(lock, [this] { return path_ready_; });
  }

  if (cap_ == nullptr || !cap_->isOpened()) {
    // Attempt reconnection
    for (int attempt = 0; attempt < reconnect_attempts_; attempt++) {
      NNDEPLOY_LOGW("Stream reconnection attempt %d/%d for %s\n",
                    attempt + 1, reconnect_attempts_, path_.c_str());
      if (cap_ != nullptr) {
        cap_->release();
        delete cap_;
        cap_ = nullptr;
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(reconnect_delay_ms_));
      cap_ = new cv::VideoCapture();
      cap_->set(cv::CAP_PROP_BUFFERSIZE, 1);
      if (cap_->open(path_, cv::CAP_FFMPEG)) {
        NNDEPLOY_LOGI("Reconnection successful on attempt %d\n", attempt + 1);
        break;
      }
      delete cap_;
      cap_ = nullptr;
    }
    if (cap_ == nullptr || !cap_->isOpened()) {
      NNDEPLOY_LOGE("All reconnection attempts failed for %s\n",
                    path_.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
  }

  cv::Mat *mat = new cv::Mat();
  bool success = cap_->read(*mat);

  if (success && !mat->empty()) {
    outputs_[0]->set(mat, false);
    index_++;
  } else {
    delete mat;
    // Stream read failed, return error to trigger reconnection on next run()
    NNDEPLOY_LOGW("Stream read failed for %s, will retry on next run\n",
                  path_.c_str());
    cap_->release();
    delete cap_;
    cap_ = nullptr;
    return base::kStatusCodeErrorInvalidParam;
  }

  return base::kStatusCodeOk;
}

// ==================== OpenCvStreamEncode ====================

base::Status OpenCvStreamEncode::init() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status OpenCvStreamEncode::deinit() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status OpenCvStreamEncode::setRefPath(const std::string &path) {
  if (ref_path_ == path) {
    return base::kStatusCodeOk;
  }
  ref_path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvStreamEncode::setPath(const std::string &path) {
  if (path_ == path) {
    return base::kStatusCodeOk;
  }
  path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status OpenCvStreamEncode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  if (mat != nullptr && !mat->empty()) {
    // For streaming encode, display frame or write to network output
    // Currently supports display via imshow for testing
#if NNDEPLOY_OS_WINDOWS || NNDEPLOY_OS_MACOS
    cv::imshow(path_, *mat);
    cv::waitKey(1);
#else
    ;
#endif
  }
  return base::kStatusCodeOk;
}

TypeCreatelDecodeRegister g_type_create_decode_node_register(
    base::kCodecTypeOpenCV, createOpenCvDecode);
TypeCreatelDecodeSharedPtrRegister
    g_type_create_decode_node_shared_ptr_register(base::kCodecTypeOpenCV,
                                                  createOpenCvDecodeSharedPtr);

Decode *createOpenCvDecode(base::CodecFlag flag, const std::string &name,
                           dag::Edge *output) {
  Decode *temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = new OpenCvImageDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagImages) {
    temp = new OpenCvImagesDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagVideo) {
    temp = new OpenCvVideoDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagCamera) {
    temp = new OpenCvCameraDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagStreaming) {
    temp = new OpenCvStreamDecode(name, {}, {output}, flag);
  }

  return temp;
}

std::shared_ptr<Decode> createOpenCvDecodeSharedPtr(base::CodecFlag flag,
                                                    const std::string &name,
                                                    dag::Edge *output) {
  std::shared_ptr<Decode> temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = std::shared_ptr<OpenCvImageDecode>(
        new OpenCvImageDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagImages) {
    temp = std::shared_ptr<OpenCvImagesDecode>(
        new OpenCvImagesDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagVideo) {
    temp = std::shared_ptr<OpenCvVideoDecode>(
        new OpenCvVideoDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagCamera) {
    temp = std::shared_ptr<OpenCvCameraDecode>(
        new OpenCvCameraDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagStreaming) {
    temp = std::shared_ptr<OpenCvStreamDecode>(
        new OpenCvStreamDecode(name, {}, {output}, flag));
  }

  return temp;
}

TypeCreatelEncodeRegister g_type_create_encode_node_register(
    base::kCodecTypeOpenCV, createOpenCvEncode);
TypeCreatelEncodeSharedPtrRegister
    g_type_create_encode_node_shared_ptr_register(base::kCodecTypeOpenCV,
                                                  createOpenCvEncodeSharedPtr);

Encode *createOpenCvEncode(base::CodecFlag flag, const std::string &name,
                           dag::Edge *input) {
  Encode *temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = new OpenCvImageEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagImages) {
    temp = new OpenCvImagesEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagVideo) {
    temp = new OpenCvVideoEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagCamera) {
    temp = new OpenCvCameraEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagStreaming) {
    temp = new OpenCvStreamEncode(name, {input}, {}, flag);
  }

  return temp;
}

std::shared_ptr<Encode> createOpenCvEncodeSharedPtr(base::CodecFlag flag,
                                                    const std::string &name,
                                                    dag::Edge *input) {
  std::shared_ptr<Encode> temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = std::shared_ptr<OpenCvImageEncode>(
        new OpenCvImageEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagImages) {
    temp = std::shared_ptr<OpenCvImagesEncode>(
        new OpenCvImagesEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagVideo) {
    temp = std::shared_ptr<OpenCvVideoEncode>(
        new OpenCvVideoEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagCamera) {
    temp = std::shared_ptr<OpenCvCameraEncode>(
        new OpenCvCameraEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagStreaming) {
    temp = std::shared_ptr<OpenCvStreamEncode>(
        new OpenCvStreamEncode(name, {input}, {}, flag));
  }

  return temp;
}

REGISTER_NODE("nndeploy::codec::OpenCvImageDecode", OpenCvImageDecode);
REGISTER_NODE("nndeploy::codec::OpenCvImagesDecode", OpenCvImagesDecode);
REGISTER_NODE("nndeploy::codec::OpenCvVideoDecode", OpenCvVideoDecode);
REGISTER_NODE("nndeploy::codec::OpenCvCameraDecode", OpenCvCameraDecode);
REGISTER_NODE("nndeploy::codec::OpenCvStreamDecode", OpenCvStreamDecode);
REGISTER_NODE("nndeploy::codec::OpenCvImageEncode", OpenCvImageEncode);
REGISTER_NODE("nndeploy::codec::OpenCvImagesEncode", OpenCvImagesEncode);
REGISTER_NODE("nndeploy::codec::OpenCvVideoEncode", OpenCvVideoEncode);
REGISTER_NODE("nndeploy::codec::OpenCvCameraEncode", OpenCvCameraEncode);
REGISTER_NODE("nndeploy::codec::OpenCvStreamEncode", OpenCvStreamEncode);

}  // namespace codec
}  // namespace nndeploy