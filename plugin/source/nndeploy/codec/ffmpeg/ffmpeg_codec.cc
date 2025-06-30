#include "nndeploy/codec/ffmpeg/ffmpeg_codec.h"

#include "nndeploy/base/file.h"

namespace nndeploy {
namespace codec {
base::Status FFmpegImageDecode::init() { 
    av_register_all()
    return base::kStatusCodeOk; 
}

base::Status FFmpegImageDecode::deinit() { 
    if (format_ctx_) {
        avformat_close_input(&format_ctx_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (frame_) {
        av_frame_free(&frame_);
    }
    return base::kStatusCodeOk; 
}

base::Status FFmpegImageDecode::setPath(const std::string &path) {
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
  return base::kStatusCodeOk;
}
}
}