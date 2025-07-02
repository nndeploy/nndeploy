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

base::Status FFmpegImageDecode::run() {
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
  // 打开图片文件
  if (avformat_open_input(&format_ctx_, path_, NULL, NULL) < 0) {
    fprintf(stderr, "Could not open input file %s\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  // 获取流信息
  if (avformat_find_stream_info(format_ctx_, NULL) < 0) {
    fprintf(stderr, "Could not find stream information\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 查找视频流（图片文件通常只有一个视频流）
  int video_stream_index = -1;
  for (int i = 0; i < format_ctx_->nb_streams; i++) {
    if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_index = i;
      break;
    }
  }

  if (video_stream_index == -1) {
    fprintf(stderr, "Could not find video stream\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  
  // 获取视频解码器
  AVCodecParameters *codecpar = format_ctx_->streams[video_stream_index]->codecpar;
  AVCodec *codec = avcodec_find_decoder(codecpar->codec_id);
  if (!codec) {
    fprintf(stderr, "Codec not found\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 打开解码器
  codec_ctx_ = avcodec_alloc_context3(codec);
  if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
    fprintf(stderr, "Failed to copy codec parameters to codec context\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  if (avcodec_open2(codec_ctx_, codec, NULL) < 0) {
    fprintf(stderr, "Failed to open codec\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 获取解码后的图像帧
  AVPacket packet;
  frame_ = av_frame_alloc();
  if (av_read_frame(format_ctx_, &packet) >= 0) {
      if (packet.stream_index == video_stream_index) {
          if (avcodec_send_packet(codec_ctx_, &packet) >= 0) {
              while (avcodec_receive_frame(codec_ctx_, frame_frame_) >= 0) {
                  // 处理解码后的图像数据
                  printf("Decoded frame: width=%d, height=%d\n", frame_->width, frame_->height);
                  // 这里可以将图像数据渲染、保存或进行其他处理
                  width_ = frame_->width;
                  height_ = frame_->height;
              }
          }
      }
      av_packet_unref(&packet);
  }

  // NNDEPLOY_LOGE("OpenCvImageDecode::run() width_[%d] height_[%d]\n",
  // width_, height_);
  outputs_[0]->set(frame_, false);
  index_++;
  return base::kStatusCodeOk;
}


}
}