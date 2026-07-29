
#include "nndeploy/codec/ffmpeg/ffmpeg_codec.h"

#include "nndeploy/base/file.h"

// Helper: apply SW codec bitrate and thread config to AVCodecContext
static void applySwBitrateAndThreads(AVCodecContext* ctx,
                                     const FFmpegSWCodecConfig& cfg) {
  if (cfg.bit_rate > 0) ctx->bit_rate = cfg.bit_rate;
  if (cfg.threads > 0) ctx->thread_count = cfg.threads;
}

// Helper: find SW decoder considering sw_codec_config
static const AVCodec* resolveDecoder(AVCodecID codec_id,
                                     const FFmpegSWCodecConfig& cfg) {
  if (!cfg.codec_name.empty()) {
    const AVCodec* c = avcodec_find_decoder_by_name(cfg.codec_name.c_str());
    if (c) return c;
  }
  return avcodec_find_decoder(codec_id);
}

// Helper: open encoder with SW codec config (preset, tune, profile, crf)
static int openEncoderWithOpts(AVCodecContext* ctx, const AVCodec* codec,
                               const FFmpegSWCodecConfig& cfg) {
  AVDictionary* opts = nullptr;
  if (!cfg.preset.empty())
    av_dict_set(&opts, "preset", cfg.preset.c_str(), 0);
  if (!cfg.tune.empty())
    av_dict_set(&opts, "tune", cfg.tune.c_str(), 0);
  if (!cfg.profile.empty())
    av_dict_set(&opts, "profile", cfg.profile.c_str(), 0);
  if (cfg.crf > 0) {
    char crf_str[8];
    snprintf(crf_str, sizeof(crf_str), "%d", cfg.crf);
    av_dict_set(&opts, "crf", crf_str, 0);
  }
  if (cfg.threads > 0) {
    char thr_str[8];
    snprintf(thr_str, sizeof(thr_str), "%d", cfg.threads);
    av_dict_set(&opts, "threads", thr_str, 0);
  }
  int ret = avcodec_open2(ctx, codec, &opts);
  av_dict_free(&opts);
  return ret;
}

namespace nndeploy {
namespace codec {

static void setupHwDecoder(FFmpegHWCodec* hw_codec, AVCodecContext* codec_ctx,
                           AVCodecID codec_id, int width, int height);

static enum AVPixelFormat get_hw_format(AVCodecContext* ctx,
                                         const enum AVPixelFormat* pix_fmts) {
  AVPixelFormat hw_pix_fmt = static_cast<AVPixelFormat>(
      reinterpret_cast<intptr_t>(ctx->opaque));
  for (const enum AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
    if (*p == hw_pix_fmt) return hw_pix_fmt;
  }
  return pix_fmts[0];
}

static void setupHwDecoder(FFmpegHWCodec* hw_codec, AVCodecContext* codec_ctx,
                           AVCodecID codec_id, int width, int height) {
  base::Status status;
  AVHWDeviceType av_type = hw_codec->initHwDevice(&status);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("setupHwDecoder: initHwDevice failed\n");
    return;
  }

  AVPixelFormat hw_pix_fmt;
  switch (av_type) {
    case AV_HWDEVICE_TYPE_CUDA:
      hw_pix_fmt = AV_PIX_FMT_CUDA;
      break;
    case AV_HWDEVICE_TYPE_QSV:
      hw_pix_fmt = AV_PIX_FMT_QSV;
      break;
    case AV_HWDEVICE_TYPE_VAAPI:
      hw_pix_fmt = AV_PIX_FMT_VAAPI;
      break;
    case AV_HWDEVICE_TYPE_VIDEOTOOLBOX:
      hw_pix_fmt = AV_PIX_FMT_VIDEOTOOLBOX;
      break;
    case AV_HWDEVICE_TYPE_V4L2M2M:
    case AV_HWDEVICE_TYPE_DRM:
      hw_pix_fmt = AV_PIX_FMT_NV12;
      break;
    default:
      hw_pix_fmt = AV_PIX_FMT_NONE;
      break;
  }

  if (hw_pix_fmt == AV_PIX_FMT_NONE) {
    NNDEPLOY_LOGE("setupHwDecoder: unsupported HW device type\n");
    return;
  }

  hw_codec->setHwPixelFormat(hw_pix_fmt);
  codec_ctx->opaque = reinterpret_cast<void*>(
      static_cast<intptr_t>(hw_pix_fmt));
  codec_ctx->get_format = get_hw_format;
  codec_ctx->hw_device_ctx = av_buffer_ref(hw_codec->getHwDeviceRef());
  if (!codec_ctx->hw_device_ctx) {
    NNDEPLOY_LOGE("setupHwDecoder: av_buffer_ref failed\n");
  }
}

// Helper function to free FFmpeg resources
static void freeFFmpegResources(AVFormatContext* fmt_ctx,
                                AVCodecContext* codec_ctx, AVFrame* frame,
                                AVPacket* pkt, SwsContext* sws_ctx) {
  if (sws_ctx) {
    sws_freeContext(sws_ctx);
  }
  if (codec_ctx) {
    avcodec_free_context(&codec_ctx);
  }
  if (frame) {
    av_frame_free(&frame);
  }
  if (pkt) {
    av_packet_free(&pkt);
  }
  if (fmt_ctx && !(fmt_ctx->oformat)) {
    avformat_close_input(&fmt_ctx);
  } else if (fmt_ctx) {
    avformat_free_context(fmt_ctx);
  }
}

// Helper function to convert AVFrame to cv::Mat using BGR color space
static bool convertFrameToMat(AVFrame* frame, cv::Mat* mat,
                              SwsContext** sws_ctx_ref,
                              AVCodecContext* codec_ctx) {
  if (!frame || !mat || !codec_ctx) {
    return false;
  }

  int width = codec_ctx->width;
  int height = codec_ctx->height;

  // Only create sws context if needed or if dimensions changed
  static int last_width = 0, last_height = 0;
  static AVPixelFormat last_format = AV_PIX_FMT_NONE;
  AVPixelFormat src_format = (AVPixelFormat)frame->format;

  if (*sws_ctx_ref && (last_width != width || last_height != height ||
                       last_format != src_format)) {
    sws_freeContext(*sws_ctx_ref);
    *sws_ctx_ref = nullptr;
  }

  if (!*sws_ctx_ref) {
    *sws_ctx_ref = sws_getContext(width, height, src_format, width, height,
                                  AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr,
                                  nullptr, nullptr);
    if (!*sws_ctx_ref) {
      NNDEPLOY_LOGE("Cannot initialize conversion context\n");
      return false;
    }
    last_width = width;
    last_height = height;
    last_format = src_format;
  }

  // Allocate output buffer
  uint8_t* dest_data[4] = {nullptr};
  int dest_linesize[4] = {0};
  av_image_alloc(dest_data, dest_linesize, width, height, AV_PIX_FMT_BGR24, 1);

  sws_scale(*sws_ctx_ref, frame->data, frame->linesize, 0, height, dest_data,
            dest_linesize);

  // Create cv::Mat from buffer (BGR format)
  *mat = cv::Mat(height, width, CV_8UC3, dest_data[0], dest_linesize[0]);
  return true;
}

// Helper function to decode one frame from FFmpeg
static bool decodeFrame(AVFormatContext* fmt_ctx, AVCodecContext* codec_ctx,
                        AVFrame* frame, AVPacket* pkt, SwsContext** sws_ctx,
                        cv::Mat* mat, int* index) {
  int ret = av_read_frame(fmt_ctx, pkt);
  if (ret < 0) {
    return false;  // End of file or error
  }

  if (pkt->stream_index != -1) {
    ret = avcodec_send_packet(codec_ctx, pkt);
    if (ret < 0) {
      av_packet_unref(pkt);
      return false;
    }

    while (ret >= 0) {
      ret = avcodec_receive_frame(codec_ctx, frame);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        break;
      } else if (ret < 0) {
        av_packet_unref(pkt);
        return false;
      }
      (*index)++;
      // Convert to cv::Mat
      if (!convertFrameToMat(frame, mat, sws_ctx, codec_ctx)) {
        av_packet_unref(pkt);
        return false;
      }
    }
  }
  av_packet_unref(pkt);
  return true;
}

// ==================== FFmpegImageDecode ====================

base::Status FFmpegImageDecode::init() { return base::kStatusCodeOk; }

base::Status FFmpegImageDecode::deinit() {
  freeFFmpegResources(fmt_ctx_, codec_ctx_, frame_, pkt_, sws_ctx_);
  fmt_ctx_ = nullptr;
  codec_ctx_ = nullptr;
  frame_ = nullptr;
  pkt_ = nullptr;
  sws_ctx_ = nullptr;
  video_stream_idx_ = -1;
  return base::kStatusCodeOk;
}

base::Status FFmpegImageDecode::setPath(const std::string& path) {
  if (!base::exists(path)) {
    NNDEPLOY_LOGE("path[%s] is not exists!\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  deinit();

  // Open input file
  if (avformat_open_input(&fmt_ctx_, path.c_str(), nullptr, nullptr) != 0) {
    NNDEPLOY_LOGE("Cannot open input file %s\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
    NNDEPLOY_LOGE("Cannot find stream info\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  // Find video stream
  video_stream_idx_ = -1;
  for (unsigned int i = 0; i < fmt_ctx_->nb_streams; i++) {
    if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_idx_ = i;
      break;
    }
  }

  if (video_stream_idx_ == -1) {
    // For single image, try to treat as video stream
    NNDEPLOY_LOGE("No video stream found in %s\n", path.c_str());
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  AVStream* stream = fmt_ctx_->streams[video_stream_idx_];
  const AVCodec* codec = resolveDecoder(stream->codecpar->codec_id,
                                        sw_codec_config_);
  if (!codec) {
    NNDEPLOY_LOGE("Codec not found\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  codec_ctx_ = avcodec_alloc_context3(codec);
  if (!codec_ctx_) {
    NNDEPLOY_LOGE("Cannot allocate codec context\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  avcodec_parameters_to_context(codec_ctx_, stream->codecpar);
  applySwBitrateAndThreads(codec_ctx_, sw_codec_config_);
  if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
    NNDEPLOY_LOGE("Cannot open codec\n");
    avcodec_free_context(&codec_ctx_);
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  frame_ = av_frame_alloc();
  pkt_ = av_packet_alloc();

  if (!frame_ || !pkt_) {
    NNDEPLOY_LOGE("Cannot allocate frame or packet\n");
    freeFFmpegResources(fmt_ctx_, codec_ctx_, frame_, pkt_, sws_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  width_ = codec_ctx_->width;
  height_ = codec_ctx_->height;
  size_ = 1;
  loop_count_ = size_;
  path_ = path;
  return base::kStatusCodeOk;
}

base::Status FFmpegImageDecode::run() {
  if (!fmt_ctx_ || !codec_ctx_ || !frame_ || !pkt_) {
    return base::kStatusCodeErrorNullParam;
  }

  if (index_ >= size_) {
    return base::kStatusCodeOk;
  }

  cv::Mat* mat = new cv::Mat();

  // Read and decode one frame
  int ret = av_read_frame(fmt_ctx_, pkt_);
  if (ret >= 0) {
    ret = avcodec_send_packet(codec_ctx_, pkt_);
    if (ret >= 0) {
      ret = avcodec_receive_frame(codec_ctx_, frame_);
      if (ret >= 0) {
        convertFrameToMat(frame_, mat, &sws_ctx_, codec_ctx_);
      }
    }
    av_packet_unref(pkt_);
  }

  outputs_[0]->set(mat, false);
  index_++;
  return base::kStatusCodeOk;
}

void FFmpegImageDecode::setHwDeviceType(base::FFmpegHWDeviceType hw_type) {
  if (hw_codec_) delete hw_codec_;
  hw_codec_ = new FFmpegHWCodec(hw_type);
}

void FFmpegImageDecode::setSwCodecConfig(const FFmpegSWCodecConfig& config) {
  sw_codec_config_ = config;
}

// ==================== FFmpegImagesDecode ====================

base::Status FFmpegImagesDecode::init() { return base::kStatusCodeOk; }

base::Status FFmpegImagesDecode::deinit() {
  freeFFmpegResources(fmt_ctx_, codec_ctx_, frame_, pkt_, sws_ctx_);
  fmt_ctx_ = nullptr;
  codec_ctx_ = nullptr;
  frame_ = nullptr;
  pkt_ = nullptr;
  sws_ctx_ = nullptr;
  video_stream_idx_ = -1;
  images_.clear();
  return base::kStatusCodeOk;
}

base::Status FFmpegImagesDecode::setPath(const std::string& path) {
  if (!base::isDirectory(path)) {
    NNDEPLOY_LOGE("path[%s] is not Directory!\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  deinit();
  index_ = 0;
  path_ = path;

  // Glob image files
  base::glob(path, "*.jpg", images_);
  std::vector<std::string> png_result;
  base::glob(path, "*.png", png_result);
  images_.insert(images_.end(), png_result.begin(), png_result.end());
  std::vector<std::string> jpeg_result;
  base::glob(path, "*.jpeg", jpeg_result);
  images_.insert(images_.end(), jpeg_result.begin(), jpeg_result.end());

  size_ = (int)images_.size();
  if (size_ == 0) {
    NNDEPLOY_LOGE("path[%s] not exist pic!\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  loop_count_ = size_;
  return base::kStatusCodeOk;
}

base::Status FFmpegImagesDecode::run() {
  if (index_ >= size_) {
    return base::kStatusCodeOk;
  }

  std::string image_path = images_[index_];
  base::Status status = setPath(image_path);
  if (status != base::kStatusCodeOk) {
    return status;
  }

  cv::Mat* mat = new cv::Mat();
  int ret = av_read_frame(fmt_ctx_, pkt_);
  if (ret >= 0) {
    ret = avcodec_send_packet(codec_ctx_, pkt_);
    if (ret >= 0) {
      ret = avcodec_receive_frame(codec_ctx_, frame_);
      if (ret >= 0) {
        convertFrameToMat(frame_, mat, &sws_ctx_, codec_ctx_);
      }
    }
    av_packet_unref(pkt_);
  }

  outputs_[0]->set(mat, false);
  index_++;
  return base::kStatusCodeOk;
}

// ==================== FFmpegVideoDecode ====================

base::Status FFmpegVideoDecode::init() { return base::kStatusCodeOk; }

base::Status FFmpegVideoDecode::deinit() {
  freeFFmpegResources(fmt_ctx_, codec_ctx_, frame_, pkt_, sws_ctx_);
  fmt_ctx_ = nullptr;
  codec_ctx_ = nullptr;
  frame_ = nullptr;
  pkt_ = nullptr;
  sws_ctx_ = nullptr;
  video_stream_idx_ = -1;
  return base::kStatusCodeOk;
}

base::Status FFmpegVideoDecode::setPath(const std::string& path) {
  bool is_url = path.find("://") != std::string::npos;

  if (!is_url && !base::exists(path)) {
    NNDEPLOY_LOGE("path[%s] is not exists!\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  deinit();
  index_ = 0;
  path_ = path;

  // Set network options for streaming protocols
  AVDictionary* options = nullptr;
  if (is_url) {
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0);
    av_dict_set(&options, "buffer_size", "1024000", 0);
  }

  if (avformat_open_input(&fmt_ctx_, path.c_str(), nullptr, &options) != 0) {
    NNDEPLOY_LOGE("Cannot open input file/stream %s\n", path.c_str());
    av_dict_free(&options);
    return base::kStatusCodeErrorInvalidParam;
  }
  av_dict_free(&options);

  if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
    NNDEPLOY_LOGE("Cannot find stream info\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  video_stream_idx_ = -1;
  for (unsigned int i = 0; i < fmt_ctx_->nb_streams; i++) {
    if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_idx_ = i;
      break;
    }
  }

  if (video_stream_idx_ == -1) {
    NNDEPLOY_LOGE("No video stream found\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  AVStream* stream = fmt_ctx_->streams[video_stream_idx_];
  const AVCodec* codec = nullptr;
  // Try HW decoder when hw_codec_ is set (user called setHwDeviceType)
  // prefer_hw_first=false overrides to force pure SW even if hw_codec_ is set
  if (sw_codec_config_.prefer_hw_first && hw_codec_ && hw_codec_->isEnabled()) {
    codec = hw_codec_->findHwDecoder(stream->codecpar->codec_id,
                                      stream->codecpar->width,
                                      stream->codecpar->height);
  }
  if (!codec) {
    codec = resolveDecoder(stream->codecpar->codec_id, sw_codec_config_);
  }
  if (!codec) {
    NNDEPLOY_LOGE("Codec not found\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  codec_ctx_ = avcodec_alloc_context3(codec);
  if (!codec_ctx_) {
    NNDEPLOY_LOGE("Cannot allocate codec context\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  avcodec_parameters_to_context(codec_ctx_, stream->codecpar);
  // Set up HW decoder acceleration before opening codec
  if (hw_codec_ && hw_codec_->isEnabled()) {
    setupHwDecoder(hw_codec_, codec_ctx_, stream->codecpar->codec_id,
                   stream->codecpar->width, stream->codecpar->height);
  }
  if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
    NNDEPLOY_LOGE("Cannot open codec\n");
    avcodec_free_context(&codec_ctx_);
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  frame_ = av_frame_alloc();
  pkt_ = av_packet_alloc();

  if (!frame_ || !pkt_) {
    NNDEPLOY_LOGE("Cannot allocate frame or packet\n");
    freeFFmpegResources(fmt_ctx_, codec_ctx_, frame_, pkt_, sws_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  size_ = (int)stream->nb_frames;
  if (size_ <= 0) {
    if (is_url) {
      size_ = INT_MAX;
    } else {
      size_ = (int)(fmt_ctx_->duration / 1000000 * 30);
    }
  }
  fps_ = av_q2d(stream->avg_frame_rate);
  width_ = codec_ctx_->width;
  height_ = codec_ctx_->height;
  loop_count_ = size_;

  return base::kStatusCodeOk;
}

base::Status FFmpegVideoDecode::run() {
  if (!fmt_ctx_ || !codec_ctx_ || !frame_ || !pkt_) {
    return base::kStatusCodeErrorNullParam;
  }

  if (size_ != INT_MAX && index_ >= size_) {
    return base::kStatusCodeOk;
  }

  cv::Mat* mat = new cv::Mat();
  int ret = av_read_frame(fmt_ctx_, pkt_);

  if (ret >= 0) {
    if (pkt_->stream_index == video_stream_idx_) {
      ret = avcodec_send_packet(codec_ctx_, pkt_);
      if (ret >= 0) {
        ret = avcodec_receive_frame(codec_ctx_, frame_);
        if (ret >= 0) {
          index_++;
          convertFrameToMat(frame_, mat, &sws_ctx_, codec_ctx_);
        }
      }
    }
    av_packet_unref(pkt_);
  } else {
    if (size_ == INT_MAX) {
      NNDEPLOY_LOGW("Network stream read failed, attempting to reconnect...\n");
      delete mat;

      deinit();
      setPath(path_);

      if (!fmt_ctx_ || !codec_ctx_) {
        NNDEPLOY_LOGE("Failed to reconnect to stream %s\n", path_.c_str());
        return base::kStatusCodeErrorInvalidParam;
      }

      cv::Mat* retry_mat = new cv::Mat();
      ret = av_read_frame(fmt_ctx_, pkt_);
      if (ret >= 0) {
        if (pkt_->stream_index == video_stream_idx_) {
          ret = avcodec_send_packet(codec_ctx_, pkt_);
          if (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx_, frame_);
            if (ret >= 0) {
              index_++;
              convertFrameToMat(frame_, retry_mat, &sws_ctx_, codec_ctx_);
            }
          }
        }
        av_packet_unref(pkt_);
      }

      outputs_[0]->set(retry_mat, false);
      return base::kStatusCodeOk;
    } else {
      delete mat;
      if (index_ == 0) {
        NNDEPLOY_LOGW("Video file has no frames\n");
      }
    }
  }

  outputs_[0]->set(mat, false);
  if (size_ != INT_MAX && index_ >= size_) {
    avformat_close_input(&fmt_ctx_);
  }
  return base::kStatusCodeOk;
}

void FFmpegVideoDecode::setHwDeviceType(base::FFmpegHWDeviceType hw_type) {
  if (hw_codec_) delete hw_codec_;
  hw_codec_ = new FFmpegHWCodec(hw_type);
}

void FFmpegVideoDecode::setSwCodecConfig(const FFmpegSWCodecConfig& config) {
  sw_codec_config_ = config;
}

// ==================== FFmpegCameraDecode ====================

base::Status FFmpegCameraDecode::init() {
  size_ = 0;
  return base::kStatusCodeOk;
}

base::Status FFmpegCameraDecode::deinit() {
  freeFFmpegResources(fmt_ctx_, codec_ctx_, frame_, pkt_, sws_ctx_);
  fmt_ctx_ = nullptr;
  codec_ctx_ = nullptr;
  frame_ = nullptr;
  pkt_ = nullptr;
  sws_ctx_ = nullptr;
  video_stream_idx_ = -1;
  return base::kStatusCodeOk;
}

base::Status FFmpegCameraDecode::setPath(const std::string& path) {
  deinit();
  index_ = 0;
  path_ = path;

  std::string url = path.empty() ? "0" : path;

  if (avformat_open_input(&fmt_ctx_, url.c_str(), nullptr, nullptr) != 0) {
    NNDEPLOY_LOGE("Cannot open camera input %s\n", url.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
    NNDEPLOY_LOGE("Cannot find stream info\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  video_stream_idx_ = -1;
  for (unsigned int i = 0; i < fmt_ctx_->nb_streams; i++) {
    if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_idx_ = i;
      break;
    }
  }

  if (video_stream_idx_ == -1) {
    NNDEPLOY_LOGE("No video stream found\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  AVStream* stream = fmt_ctx_->streams[video_stream_idx_];
  const AVCodec* codec = resolveDecoder(stream->codecpar->codec_id,
                                        sw_codec_config_);
  if (!codec) {
    NNDEPLOY_LOGE("Codec not found\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  codec_ctx_ = avcodec_alloc_context3(codec);
  if (!codec_ctx_) {
    NNDEPLOY_LOGE("Cannot allocate codec context\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  avcodec_parameters_to_context(codec_ctx_, stream->codecpar);
  if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
    NNDEPLOY_LOGE("Cannot open codec\n");
    avcodec_free_context(&codec_ctx_);
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  frame_ = av_frame_alloc();
  pkt_ = av_packet_alloc();

  if (!frame_ || !pkt_) {
    NNDEPLOY_LOGE("Cannot allocate frame or packet\n");
    freeFFmpegResources(fmt_ctx_, codec_ctx_, frame_, pkt_, sws_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  size_ = INT_MAX;
  fps_ = av_q2d(stream->avg_frame_rate);
  width_ = codec_ctx_->width;
  height_ = codec_ctx_->height;
  loop_count_ = size_;

  return base::kStatusCodeOk;
}

base::Status FFmpegCameraDecode::run() {
  if (!fmt_ctx_ || !codec_ctx_ || !frame_ || !pkt_) {
    return base::kStatusCodeErrorNullParam;
  }

  if (index_ >= size_) {
    return base::kStatusCodeOk;
  }

  cv::Mat* mat = new cv::Mat();
  int ret = av_read_frame(fmt_ctx_, pkt_);
  if (ret >= 0) {
    if (pkt_->stream_index == video_stream_idx_) {
      ret = avcodec_send_packet(codec_ctx_, pkt_);
      if (ret >= 0) {
        ret = avcodec_receive_frame(codec_ctx_, frame_);
        if (ret >= 0) {
          index_++;
          convertFrameToMat(frame_, mat, &sws_ctx_, codec_ctx_);
        }
      }
    }
    av_packet_unref(pkt_);
  }

  outputs_[0]->set(mat, false);
  return base::kStatusCodeOk;
}

void FFmpegCameraDecode::setHwDeviceType(base::FFmpegHWDeviceType hw_type) {
  if (hw_codec_) delete hw_codec_;
  hw_codec_ = new FFmpegHWCodec(hw_type);
}

void FFmpegCameraDecode::setSwCodecConfig(const FFmpegSWCodecConfig& config) {
  sw_codec_config_ = config;
}

// ==================== FFmpegStreamDecode ====================

base::Status FFmpegStreamDecode::init() { return base::kStatusCodeOk; }

base::Status FFmpegStreamDecode::deinit() {
  freeFFmpegResources(fmt_ctx_, codec_ctx_, frame_, pkt_, sws_ctx_);
  fmt_ctx_ = nullptr;
  codec_ctx_ = nullptr;
  frame_ = nullptr;
  pkt_ = nullptr;
  sws_ctx_ = nullptr;
  video_stream_idx_ = -1;
  return base::kStatusCodeOk;
}

base::Status FFmpegStreamDecode::setPath(const std::string& path) {
  deinit();
  index_ = 0;
  path_ = path;

  // Set network streaming options: TCP transport, longer timeout, larger buffer
  AVDictionary* options = nullptr;
  av_dict_set(&options, "rtsp_transport", "tcp", 0);
  av_dict_set(&options, "stimeout", "5000000", 0);
  av_dict_set(&options, "buffer_size", "2048000", 0);
  av_dict_set(&options, "reconnect", "1", 0);
  av_dict_set(&options, "reconnect_at_eof", "1", 0);
  av_dict_set(&options, "reconnect_streamed", "1", 0);
  av_dict_set(&options, "reconnect_delay_max", "5", 0);

  if (avformat_open_input(&fmt_ctx_, path.c_str(), nullptr, &options) != 0) {
    NNDEPLOY_LOGE("Cannot open stream %s\n", path.c_str());
    av_dict_free(&options);
    return base::kStatusCodeErrorInvalidParam;
  }
  av_dict_free(&options);

  if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
    NNDEPLOY_LOGE("Cannot find stream info\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  video_stream_idx_ = -1;
  for (unsigned int i = 0; i < fmt_ctx_->nb_streams; i++) {
    if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_idx_ = i;
      break;
    }
  }

  if (video_stream_idx_ == -1) {
    NNDEPLOY_LOGE("No video stream found\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  AVStream* stream = fmt_ctx_->streams[video_stream_idx_];
  const AVCodec* codec = nullptr;
  if (sw_codec_config_.prefer_hw_first && hw_codec_ && hw_codec_->isEnabled()) {
    codec = hw_codec_->findHwDecoder(stream->codecpar->codec_id,
                                      stream->codecpar->width,
                                      stream->codecpar->height);
  }
  if (!codec) {
    codec = resolveDecoder(stream->codecpar->codec_id, sw_codec_config_);
  }
  if (!codec) {
    NNDEPLOY_LOGE("Codec not found\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  codec_ctx_ = avcodec_alloc_context3(codec);
  if (!codec_ctx_) {
    NNDEPLOY_LOGE("Cannot allocate codec context\n");
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  avcodec_parameters_to_context(codec_ctx_, stream->codecpar);
  if (hw_codec_ && hw_codec_->isEnabled()) {
    setupHwDecoder(hw_codec_, codec_ctx_, stream->codecpar->codec_id,
                   stream->codecpar->width, stream->codecpar->height);
  }
  if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
    NNDEPLOY_LOGE("Cannot open codec\n");
    avcodec_free_context(&codec_ctx_);
    avformat_close_input(&fmt_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  frame_ = av_frame_alloc();
  pkt_ = av_packet_alloc();

  if (!frame_ || !pkt_) {
    NNDEPLOY_LOGE("Cannot allocate frame or packet\n");
    freeFFmpegResources(fmt_ctx_, codec_ctx_, frame_, pkt_, sws_ctx_);
    return base::kStatusCodeErrorInvalidParam;
  }

  size_ = INT_MAX;
  fps_ = av_q2d(stream->avg_frame_rate);
  width_ = codec_ctx_->width;
  height_ = codec_ctx_->height;
  loop_count_ = size_;

  return base::kStatusCodeOk;
}

base::Status FFmpegStreamDecode::run() {
  if (!fmt_ctx_ || !codec_ctx_ || !frame_ || !pkt_) {
    return base::kStatusCodeErrorNullParam;
  }

  if (index_ >= size_) {
    return base::kStatusCodeOk;
  }

  cv::Mat* mat = new cv::Mat();
  int ret = av_read_frame(fmt_ctx_, pkt_);

  if (ret >= 0) {
    if (pkt_->stream_index == video_stream_idx_) {
      ret = avcodec_send_packet(codec_ctx_, pkt_);
      if (ret >= 0) {
        ret = avcodec_receive_frame(codec_ctx_, frame_);
        if (ret >= 0) {
          index_++;
          convertFrameToMat(frame_, mat, &sws_ctx_, codec_ctx_);
        }
      }
    }
    av_packet_unref(pkt_);
  } else {
    // Stream read failed — attempt reconnection
    NNDEPLOY_LOGW("Stream read failed, attempting reconnection...\n");
    delete mat;

    for (int attempt = 1; attempt <= reconnect_attempts_; ++attempt) {
      NNDEPLOY_LOGI("Reconnection attempt %d/%d\n", attempt, reconnect_attempts_);
      std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms_));

      deinit();
      if (setPath(path_) == base::kStatusCodeOk) {
        break;
      }
    }

    if (!fmt_ctx_ || !codec_ctx_) {
      NNDEPLOY_LOGE("All reconnection attempts failed for %s\n", path_.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }

    cv::Mat* retry_mat = new cv::Mat();
    ret = av_read_frame(fmt_ctx_, pkt_);
    if (ret >= 0) {
      if (pkt_->stream_index == video_stream_idx_) {
        ret = avcodec_send_packet(codec_ctx_, pkt_);
        if (ret >= 0) {
          ret = avcodec_receive_frame(codec_ctx_, frame_);
          if (ret >= 0) {
            index_++;
            convertFrameToMat(frame_, retry_mat, &sws_ctx_, codec_ctx_);
          }
        }
      }
      av_packet_unref(pkt_);
    }
    outputs_[0]->set(retry_mat, false);
    return base::kStatusCodeOk;
  }

  outputs_[0]->set(mat, false);
  return base::kStatusCodeOk;
}

void FFmpegStreamDecode::setHwDeviceType(base::FFmpegHWDeviceType hw_type) {
  if (hw_codec_) delete hw_codec_;
  hw_codec_ = new FFmpegHWCodec(hw_type);
}

void FFmpegStreamDecode::setSwCodecConfig(const FFmpegSWCodecConfig& config) {
  sw_codec_config_ = config;
}

// ==================== FFmpegStreamEncode ====================

base::Status FFmpegStreamEncode::init() {
  initialized_ = false;
  return base::kStatusCodeOk;
}

base::Status FFmpegStreamEncode::deinit() {
  if (codec_ctx_) {
    avcodec_free_context(&codec_ctx_);
    codec_ctx_ = nullptr;
  }
  if (fmt_ctx_) {
    avformat_close_input(&fmt_ctx_);
    fmt_ctx_ = nullptr;
  }
  if (frame_) {
    av_frame_free(&frame_);
    frame_ = nullptr;
  }
  if (pkt_) {
    av_packet_free(&pkt_);
    pkt_ = nullptr;
  }
  if (sws_ctx_) {
    sws_freeContext(sws_ctx_);
    sws_ctx_ = nullptr;
  }
  initialized_ = false;
  return base::kStatusCodeOk;
}

base::Status FFmpegStreamEncode::setRefPath(const std::string& ref_path) {
  ref_path_ = ref_path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status FFmpegStreamEncode::setPath(const std::string& path) {
  path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status FFmpegStreamEncode::run() {
  cv::Mat* mat = inputs_[0]->getCvMat(this);
  if (!mat || mat->empty()) {
    NNDEPLOY_LOGE("Input mat is empty\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  return FFmpegVideoEncode::run();
}

void FFmpegStreamEncode::setHwDeviceType(base::FFmpegHWDeviceType hw_type) {
  if (hw_codec_) delete hw_codec_;
  hw_codec_ = new FFmpegHWCodec(hw_type);
}

void FFmpegStreamEncode::setSwCodecConfig(const FFmpegSWCodecConfig& config) {
  sw_codec_config_ = config;
}

// ==================== FFmpegImageEncode ====================

base::Status FFmpegImageEncode::init() {
  initialized_ = false;
  return base::kStatusCodeOk;
}

base::Status FFmpegImageEncode::deinit() {
  if (codec_ctx_) {
    avcodec_free_context(&codec_ctx_);
    codec_ctx_ = nullptr;
  }
  if (frame_) {
    av_frame_free(&frame_);
    frame_ = nullptr;
  }
  if (pkt_) {
    av_packet_free(&pkt_);
    pkt_ = nullptr;
  }
  if (sws_ctx_) {
    sws_freeContext(sws_ctx_);
    sws_ctx_ = nullptr;
  }
  initialized_ = false;
  return base::kStatusCodeOk;
}

base::Status FFmpegImageEncode::setRefPath(const std::string& ref_path) {
  ref_path_ = ref_path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status FFmpegImageEncode::setPath(const std::string& path) {
  path_ = path;
  path_changed_ = true;
  size_ = 1;
  return base::kStatusCodeOk;
}

base::Status FFmpegImageEncode::run() {
  cv::Mat* mat = inputs_[0]->getCvMat(this);
  if (!mat || mat->empty()) {
    NNDEPLOY_LOGE("Input mat is empty\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  if (!initialized_ || path_changed_) {
    deinit();

    // Determine output format from file extension
    AVOutputFormat* fmt = nullptr;
    if (path_.find(".jpg") != std::string::npos ||
        path_.find(".jpeg") != std::string::npos) {
      fmt = av_guess_format("mjpeg", nullptr, nullptr);
    } else if (path_.find(".png") != std::string::npos) {
      fmt = av_guess_format("png", nullptr, nullptr);
    } else {
      fmt = av_guess_format("mjpeg", nullptr, nullptr);
    }

    if (!fmt) {
      NNDEPLOY_LOGE("Cannot find output format\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    if (avformat_alloc_output_context2(&fmt_ctx_, fmt, nullptr, path_.c_str()) <
        0) {
      NNDEPLOY_LOGE("Cannot allocate output context\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    const AVCodec* codec = avcodec_find_encoder(fmt->video_codec);
    if (!codec) {
      NNDEPLOY_LOGE("Codec not found\n");
      avformat_free_context(fmt_ctx_);
      return base::kStatusCodeErrorInvalidParam;
    }

    AVStream* stream = avformat_new_stream(fmt_ctx_, codec);
    if (!stream) {
      NNDEPLOY_LOGE("Cannot create new stream\n");
      avformat_free_context(fmt_ctx_);
      return base::kStatusCodeErrorInvalidParam;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      NNDEPLOY_LOGE("Cannot allocate codec context\n");
      avformat_free_context(fmt_ctx_);
      return base::kStatusCodeErrorInvalidParam;
    }

    codec_ctx_->width = mat->cols;
    codec_ctx_->height = mat->rows;
    codec_ctx_->pix_fmt = AV_PIX_FMT_YUVJ420P;
    codec_ctx_->time_base = {1, 25};

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
      codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
      NNDEPLOY_LOGE("Cannot open codec\n");
      avcodec_free_context(&codec_ctx_);
      avformat_free_context(fmt_ctx_);
      return base::kStatusCodeErrorInvalidParam;
    }

    avcodec_parameters_from_context(stream->codecpar, codec_ctx_);

    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
      if (avio_open(&fmt_ctx_->pb, path_.c_str(), AVIO_FLAG_WRITE) < 0) {
        NNDEPLOY_LOGE("Cannot open output file\n");
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(fmt_ctx_);
        return base::kStatusCodeErrorInvalidParam;
      }
    }

    avformat_write_header(fmt_ctx_, nullptr);

    frame_ = av_frame_alloc();
    pkt_ = av_packet_alloc();
    frame_->width = codec_ctx_->width;
    frame_->height = codec_ctx_->height;
    frame_->format = codec_ctx_->pix_fmt;
    av_frame_get_buffer(frame_, 0);

    sws_ctx_ = sws_getContext(mat->cols, mat->rows, AV_PIX_FMT_BGR24,
                              codec_ctx_->width, codec_ctx_->height,
                              codec_ctx_->pix_fmt, SWS_BILINEAR, nullptr,
                              nullptr, nullptr);

    initialized_ = true;
    path_changed_ = false;
  }

  // Convert cv::Mat (BGR) to YUV frame
  uint8_t* src_data[4] = {mat->data, nullptr, nullptr, nullptr};
  int src_linesize[4] = {mat->step[0], 0, 0, 0};
  sws_scale(sws_ctx_, src_data, src_linesize, 0, mat->rows, frame_->data,
            frame_->linesize);
  frame_->pts = 0;

  int ret = avcodec_send_frame(codec_ctx_, frame_);
  if (ret < 0) {
    NNDEPLOY_LOGE("Error sending frame to encoder\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  while (ret >= 0) {
    ret = avcodec_receive_packet(codec_ctx_, pkt_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    } else if (ret < 0) {
      NNDEPLOY_LOGE("Error encoding frame\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    pkt_->stream_index = 0;
    av_interleaved_write_frame(fmt_ctx_, pkt_);
    av_packet_unref(pkt_);
  }

  return base::kStatusCodeOk;
}

// ==================== FFmpegImagesEncode ====================

base::Status FFmpegImagesEncode::init() {
  initialized_ = false;
  index_ = 0;
  return base::kStatusCodeOk;
}

base::Status FFmpegImagesEncode::deinit() {
  if (initialized_ && fmt_ctx_) {
    av_write_trailer(fmt_ctx_);
    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
      avio_closep(&fmt_ctx_->pb);
    }
  }
  if (codec_ctx_) {
    avcodec_free_context(&codec_ctx_);
    codec_ctx_ = nullptr;
  }
  if (frame_) {
    av_frame_free(&frame_);
    frame_ = nullptr;
  }
  if (pkt_) {
    av_packet_free(&pkt_);
    pkt_ = nullptr;
  }
  if (sws_ctx_) {
    sws_freeContext(sws_ctx_);
    sws_ctx_ = nullptr;
  }
  if (fmt_ctx_) {
    avformat_free_context(fmt_ctx_);
    fmt_ctx_ = nullptr;
  }
  initialized_ = false;
  return base::kStatusCodeOk;
}

base::Status FFmpegImagesEncode::setRefPath(const std::string& ref_path) {
  ref_path_ = ref_path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status FFmpegImagesEncode::setPath(const std::string& path) {
  path_ = path;
  index_ = 0;
  if (!base::isDirectory(path)) {
    NNDEPLOY_LOGE("path[%s] is not Directory!\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status FFmpegImagesEncode::run() {
  cv::Mat* mat = inputs_[0]->getCvMat(this);
  if (!mat || mat->empty()) {
    NNDEPLOY_LOGE("Input mat is empty\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  std::string full_path =
      base::joinPath(path_, std::to_string(index_) + ".jpg");

  // Encode single image
  AVOutputFormat* fmt = av_guess_format("mjpeg", nullptr, nullptr);
  AVFormatContext* local_fmt_ctx = nullptr;

  if (avformat_alloc_output_context2(&local_fmt_ctx, fmt, nullptr,
                                     full_path.c_str()) < 0) {
    return base::kStatusCodeErrorInvalidParam;
  }

  const AVCodec* codec = avcodec_find_encoder(fmt->video_codec);
  if (!codec) {
    avformat_free_context(local_fmt_ctx);
    return base::kStatusCodeErrorInvalidParam;
  }

  AVStream* stream = avformat_new_stream(local_fmt_ctx, codec);
  AVCodecContext* local_codec_ctx = avcodec_alloc_context3(codec);
  local_codec_ctx->width = mat->cols;
  local_codec_ctx->height = mat->rows;
  local_codec_ctx->pix_fmt = AV_PIX_FMT_YUVJ420P;
  local_codec_ctx->time_base = {1, 25};

  if (local_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
    local_codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  avcodec_open2(local_codec_ctx, codec, nullptr);
  avcodec_parameters_from_context(stream->codecpar, local_codec_ctx);

  if (!(local_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    avio_open(&local_fmt_ctx->pb, full_path.c_str(), AVIO_FLAG_WRITE);
  }

  avformat_write_header(local_fmt_ctx, nullptr);

  AVFrame* local_frame = av_frame_alloc();
  local_frame->width = local_codec_ctx->width;
  local_frame->height = local_codec_ctx->height;
  local_frame->format = local_codec_ctx->pix_fmt;
  av_frame_get_buffer(local_frame, 0);

  SwsContext* local_sws = sws_getContext(
      mat->cols, mat->rows, AV_PIX_FMT_BGR24, local_codec_ctx->width,
      local_codec_ctx->height, local_codec_ctx->pix_fmt, SWS_BILINEAR, nullptr,
      nullptr, nullptr);

  uint8_t* src_data[4] = {mat->data, nullptr, nullptr, nullptr};
  int src_linesize[4] = {mat->step[0], 0, 0, 0};
  sws_scale(local_sws, src_data, src_linesize, 0, mat->rows, local_frame->data,
            local_frame->linesize);
  local_frame->pts = 0;

  AVPacket* local_pkt = av_packet_alloc();
  int ret = avcodec_send_frame(local_codec_ctx, local_frame);
  while (ret >= 0) {
    ret = avcodec_receive_packet(local_codec_ctx, local_pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
    av_interleaved_write_frame(local_fmt_ctx, local_pkt);
    av_packet_unref(local_pkt);
  }

  av_write_trailer(local_fmt_ctx);
  avio_closep(&local_fmt_ctx->pb);
  sws_freeContext(local_sws);
  av_frame_free(&local_frame);
  av_packet_free(&local_pkt);
  avcodec_free_context(&local_codec_ctx);
  avformat_free_context(local_fmt_ctx);

  index_++;
  return base::kStatusCodeOk;
}

// ==================== FFmpegVideoEncode ====================

base::Status FFmpegVideoEncode::init() {
  initialized_ = false;
  return base::kStatusCodeOk;
}

base::Status FFmpegVideoEncode::deinit() {
  if (initialized_ && fmt_ctx_) {
    // Flush encoder
    int ret = avcodec_send_frame(codec_ctx_, nullptr);
    while (ret >= 0) {
      ret = avcodec_receive_packet(codec_ctx_, pkt_);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
      pkt_->stream_index = 0;
      av_interleaved_write_frame(fmt_ctx_, pkt_);
      av_packet_unref(pkt_);
    }

    av_write_trailer(fmt_ctx_);
    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
      avio_closep(&fmt_ctx_->pb);
    }
  }
  if (codec_ctx_) {
    avcodec_free_context(&codec_ctx_);
    codec_ctx_ = nullptr;
  }
  if (frame_) {
    av_frame_free(&frame_);
    frame_ = nullptr;
  }
  if (pkt_) {
    av_packet_free(&pkt_);
    pkt_ = nullptr;
  }
  if (sws_ctx_) {
    sws_freeContext(sws_ctx_);
    sws_ctx_ = nullptr;
  }
  if (fmt_ctx_) {
    avformat_free_context(fmt_ctx_);
    fmt_ctx_ = nullptr;
  }
  initialized_ = false;
  return base::kStatusCodeOk;
}

base::Status FFmpegVideoEncode::setRefPath(const std::string& ref_path) {
  ref_path_ = ref_path;
  path_changed_ = true;
  if (base::exists(ref_path)) {
    // Read reference video parameters
    AVFormatContext* ref_ctx = nullptr;
    if (avformat_open_input(&ref_ctx, ref_path.c_str(), nullptr, nullptr) >=
        0) {
      avformat_find_stream_info(ref_ctx, nullptr);
      for (unsigned int i = 0; i < ref_ctx->nb_streams; i++) {
        if (ref_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
          fps_ = av_q2d(ref_ctx->streams[i]->avg_frame_rate);
          width_ = ref_ctx->streams[i]->codecpar->width;
          height_ = ref_ctx->streams[i]->codecpar->height;
          size_ = (int)ref_ctx->streams[i]->nb_frames;
          if (size_ <= 0) {
            size_ = (int)(ref_ctx->duration / 1000000 * fps_);
          }
          break;
        }
      }
      avformat_close_input(&ref_ctx);
    }
  }
  return base::kStatusCodeOk;
}

base::Status FFmpegVideoEncode::setPath(const std::string& path) {
  if (initialized_) {
    deinit();
  }
  path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status FFmpegVideoEncode::run() {
  if (index_ >= size_ && size_ > 0) {
    return base::kStatusCodeOk;
  }

  cv::Mat* mat = inputs_[0]->getCvMat(this);
  if (!mat || mat->empty()) {
    NNDEPLOY_LOGE("Input mat is empty\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  if (!initialized_) {
    // Determine output format and codec
    AVOutputFormat* fmt = av_guess_format("mp4", nullptr, nullptr);
    if (!fmt) fmt = av_guess_format("mkv", nullptr, nullptr);
    if (!fmt) fmt = av_guess_format("avi", nullptr, nullptr);

    if (avformat_alloc_output_context2(&fmt_ctx_, fmt, nullptr, path_.c_str()) <
        0) {
      NNDEPLOY_LOGE("Cannot allocate output context\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    const AVCodec* codec = nullptr;
    if (!fourcc_.empty()) {
      // Try to find codec by fourcc
      if (fourcc_ == "h264" || fourcc_ == "avc1") {
        codec = avcodec_find_encoder_by_name("libx264");
      } else if (fourcc_ == "hevc" || fourcc_ == "hvc1") {
        codec = avcodec_find_encoder_by_name("libx265");
      } else {
        codec = avcodec_find_encoder(fmt->video_codec);
      }
    } else {
      codec = avcodec_find_encoder_by_name("libx264");
    }
    if (!codec) codec = avcodec_find_encoder(fmt->video_codec);
    if (!codec) {
      NNDEPLOY_LOGE("Codec not found\n");
      avformat_free_context(fmt_ctx_);
      return base::kStatusCodeErrorInvalidParam;
    }

    AVStream* stream = avformat_new_stream(fmt_ctx_, codec);
    if (!stream) {
      NNDEPLOY_LOGE("Cannot create new stream\n");
      avformat_free_context(fmt_ctx_);
      return base::kStatusCodeErrorInvalidParam;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      NNDEPLOY_LOGE("Cannot allocate codec context\n");
      avformat_free_context(fmt_ctx_);
      return base::kStatusCodeErrorInvalidParam;
    }

    // Set video parameters
    if (width_ == 0) width_ = mat->cols;
    if (height_ == 0) height_ = mat->rows;
    if (fps_ == 0.0) fps_ = 25.0;

    codec_ctx_->width = width_;
    codec_ctx_->height = height_;
    codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx_->time_base = {1, (int)fps_};
    codec_ctx_->bit_rate = sw_codec_config_.bit_rate > 0
                               ? sw_codec_config_.bit_rate
                               : 2000000;
    codec_ctx_->gop_size = 12;

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
      codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (openEncoderWithOpts(codec_ctx_, codec, sw_codec_config_) < 0) {
      NNDEPLOY_LOGE("Cannot open codec\n");
      avcodec_free_context(&codec_ctx_);
      avformat_free_context(fmt_ctx_);
      return base::kStatusCodeErrorInvalidParam;
    }

    avcodec_parameters_from_context(stream->codecpar, codec_ctx_);

    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
      if (avio_open(&fmt_ctx_->pb, path_.c_str(), AVIO_FLAG_WRITE) < 0) {
        NNDEPLOY_LOGE("Cannot open output file\n");
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(fmt_ctx_);
        return base::kStatusCodeErrorInvalidParam;
      }
    }

    avformat_write_header(fmt_ctx_, nullptr);

    frame_ = av_frame_alloc();
    pkt_ = av_packet_alloc();
    frame_->width = codec_ctx_->width;
    frame_->height = codec_ctx_->height;
    frame_->format = codec_ctx_->pix_fmt;
    av_frame_get_buffer(frame_, 0);

    sws_ctx_ = sws_getContext(mat->cols, mat->rows, AV_PIX_FMT_BGR24,
                              codec_ctx_->width, codec_ctx_->height,
                              codec_ctx_->pix_fmt, SWS_BILINEAR, nullptr,
                              nullptr, nullptr);

    initialized_ = true;
  }

  // Convert cv::Mat (BGR) to YUV frame
  uint8_t* src_data[4] = {mat->data, nullptr, nullptr, nullptr};
  int src_linesize[4] = {mat->step[0], 0, 0, 0};
  sws_scale(sws_ctx_, src_data, src_linesize, 0, mat->rows, frame_->data,
            frame_->linesize);
  frame_->pts = (index_) * 1000000 / fps_;

  int ret = avcodec_send_frame(codec_ctx_, frame_);
  if (ret < 0) {
    NNDEPLOY_LOGE("Error sending frame to encoder\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  while (ret >= 0) {
    ret = avcodec_receive_packet(codec_ctx_, pkt_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    } else if (ret < 0) {
      NNDEPLOY_LOGE("Error encoding frame\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    pkt_->stream_index = 0;
    pkt_->pts =
        av_rescale_q(pkt_->pts, {1, 1000000}, fmt_ctx_->streams[0]->time_base);
    pkt_->dts =
        av_rescale_q(pkt_->dts, {1, 1000000}, fmt_ctx_->streams[0]->time_base);
    av_interleaved_write_frame(fmt_ctx_, pkt_);
    av_packet_unref(pkt_);
  }

  index_++;
  return base::kStatusCodeOk;
}

void FFmpegVideoEncode::setHwDeviceType(base::FFmpegHWDeviceType hw_type) {
  if (hw_codec_) delete hw_codec_;
  hw_codec_ = new FFmpegHWCodec(hw_type);
}

void FFmpegVideoEncode::setSwCodecConfig(const FFmpegSWCodecConfig& config) {
  sw_codec_config_ = config;
}

// ==================== FFmpegCameraEncode ====================

base::Status FFmpegCameraEncode::init() {
  initialized_ = false;
  return base::kStatusCodeOk;
}

base::Status FFmpegCameraEncode::deinit() {
  return FFmpegVideoEncode::deinit();
}

base::Status FFmpegCameraEncode::setRefPath(const std::string& ref_path) {
  ref_path_ = ref_path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status FFmpegCameraEncode::setPath(const std::string& path) {
  path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status FFmpegCameraEncode::run() {
  cv::Mat* mat = inputs_[0]->getCvMat(this);
  if (!mat || mat->empty()) {
    NNDEPLOY_LOGE("Input mat is empty\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  // For camera encode, just display frame
  // (camera encode is similar to video encode, but for real-time output)
  return FFmpegVideoEncode::run();
}

void FFmpegCameraEncode::setHwDeviceType(base::FFmpegHWDeviceType hw_type) {
  if (hw_codec_) delete hw_codec_;
  hw_codec_ = new FFmpegHWCodec(hw_type);
}

void FFmpegCameraEncode::setSwCodecConfig(const FFmpegSWCodecConfig& config) {
  sw_codec_config_ = config;
}

// ==================== Factory Registration ====================

TypeCreatelDecodeRegister g_type_create_decode_node_register(
    base::kCodecTypeFFmpeg, createFFmpegDecode);
TypeCreatelDecodeSharedPtrRegister
    g_type_create_decode_node_shared_ptr_register(base::kCodecTypeFFmpeg,
                                                  createFFmpegDecodeSharedPtr);

Decode* createFFmpegDecode(base::CodecFlag flag, const std::string& name,
                           dag::Edge* output) {
  Decode* temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = new FFmpegImageDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagImages) {
    temp = new FFmpegImagesDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagVideo) {
    temp = new FFmpegVideoDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagCamera) {
    temp = new FFmpegCameraDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagStreaming) {
    temp = new FFmpegStreamDecode(name, {}, {output}, flag);
  }
  return temp;
}

std::shared_ptr<Decode> createFFmpegDecodeSharedPtr(base::CodecFlag flag,
                                                    const std::string& name,
                                                    dag::Edge* output) {
  std::shared_ptr<Decode> temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = std::shared_ptr<FFmpegImageDecode>(
        new FFmpegImageDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagImages) {
    temp = std::shared_ptr<FFmpegImagesDecode>(
        new FFmpegImagesDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagVideo) {
    temp = std::shared_ptr<FFmpegVideoDecode>(
        new FFmpegVideoDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagCamera) {
    temp = std::shared_ptr<FFmpegCameraDecode>(
        new FFmpegCameraDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagStreaming) {
    temp = std::shared_ptr<FFmpegStreamDecode>(
        new FFmpegStreamDecode(name, {}, {output}, flag));
  }
  return temp;
}

TypeCreatelEncodeRegister g_type_create_encode_node_register(
    base::kCodecTypeFFmpeg, createFFmpegEncode);
TypeCreatelEncodeSharedPtrRegister
    g_type_create_encode_node_shared_ptr_register(base::kCodecTypeFFmpeg,
                                                  createFFmpegEncodeSharedPtr);

Encode* createFFmpegEncode(base::CodecFlag flag, const std::string& name,
                           dag::Edge* input) {
  Encode* temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = new FFmpegImageEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagImages) {
    temp = new FFmpegImagesEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagVideo) {
    temp = new FFmpegVideoEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagCamera) {
    temp = new FFmpegCameraEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagStreaming) {
    temp = new FFmpegStreamEncode(name, {input}, {}, flag);
  }
  return temp;
}

std::shared_ptr<Encode> createFFmpegEncodeSharedPtr(base::CodecFlag flag,
                                                    const std::string& name,
                                                    dag::Edge* input) {
  std::shared_ptr<Encode> temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = std::shared_ptr<FFmpegImageEncode>(
        new FFmpegImageEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagImages) {
    temp = std::shared_ptr<FFmpegImagesEncode>(
        new FFmpegImagesEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagVideo) {
    temp = std::shared_ptr<FFmpegVideoEncode>(
        new FFmpegVideoEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagCamera) {
    temp = std::shared_ptr<FFmpegCameraEncode>(
        new FFmpegCameraEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagStreaming) {
    temp = std::shared_ptr<FFmpegStreamEncode>(
        new FFmpegStreamEncode(name, {input}, {}, flag));
  }
  return temp;
}

REGISTER_NODE("nndeploy::codec::FFmpegImageDecode", FFmpegImageDecode);
REGISTER_NODE("nndeploy::codec::FFmpegImagesDecode", FFmpegImagesDecode);
REGISTER_NODE("nndeploy::codec::FFmpegVideoDecode", FFmpegVideoDecode);
REGISTER_NODE("nndeploy::codec::FFmpegCameraDecode", FFmpegCameraDecode);
REGISTER_NODE("nndeploy::codec::FFmpegImageEncode", FFmpegImageEncode);
REGISTER_NODE("nndeploy::codec::FFmpegImagesEncode", FFmpegImagesEncode);
REGISTER_NODE("nndeploy::codec::FFmpegVideoEncode", FFmpegVideoEncode);
REGISTER_NODE("nndeploy::codec::FFmpegCameraEncode", FFmpegCameraEncode);
REGISTER_NODE("nndeploy::codec::FFmpegStreamDecode", FFmpegStreamDecode);
REGISTER_NODE("nndeploy::codec::FFmpegStreamEncode", FFmpegStreamEncode);

}  // namespace codec
}  // namespace nndeploy
