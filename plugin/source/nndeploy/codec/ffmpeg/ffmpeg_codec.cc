#include "nndeploy/codec/ffmpeg/ffmpeg_codec.h"

#include <algorithm>
#include <cctype>
#include <fstream>

#include "nndeploy/base/file.h"

namespace nndeploy {
namespace codec {
namespace {
std::string ToLowerCase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

AVCodecID GuessImageCodec(const std::string &path) {
  auto pos = path.find_last_of('.');
  if (pos == std::string::npos) {
    return AV_CODEC_ID_NONE;
  }
  std::string ext = ToLowerCase(path.substr(pos));
  if (ext == ".png") {
    return AV_CODEC_ID_PNG;
  }
  if (ext == ".bmp") {
    return AV_CODEC_ID_BMP;
  }
  if (ext == ".jpg" || ext == ".jpeg") {
    return AV_CODEC_ID_MJPEG;
  }
  if (ext == ".webp") {
    return AV_CODEC_ID_WEBP;
  }
  if (ext == ".tif" || ext == ".tiff") {
    return AV_CODEC_ID_TIFF;
  }
  return AV_CODEC_ID_NONE;
}

bool IsPixelFormatSupported(const AVCodec *codec, AVPixelFormat format) {
  if (codec == nullptr || codec->pix_fmts == nullptr) {
    return true;
  }
  for (const AVPixelFormat *pix_fmt = codec->pix_fmts;
       pix_fmt != nullptr && *pix_fmt != AV_PIX_FMT_NONE; ++pix_fmt) {
    if (*pix_fmt == format) {
      return true;
    }
  }
  return false;
}

}  // namespace

base::Status FFmpegImageDecode::init() { return base::kStatusCodeOk; }
base::Status FFmpegImageDecode::deinit() { return base::kStatusCodeOk; }

base::Status FFmpegImageDecode::setPath(const std::string &path) {
  if (!base::exists(path)) {
    NNDEPLOY_LOGE("path[%s] is not exists!\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  if (parallel_type_ == base::kParallelTypePipeline) {
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      path_ = path;
      path_changed_ = true;
      path_ready_ = true;
    }
    path_cv_.notify_one();
  } else {
    path_ = path;
    path_changed_ = true;
    path_ready_ = true;
  }
  if (size_ < 1) {
    size_ = 1;
  }
  this->setLoopCount(size_);
  return base::kStatusCodeOk;
}

base::Status FFmpegImageDecode::run() {
  if (index_ == 0 && parallel_type_ == base::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(path_mutex_);
    path_cv_.wait(lock, [this] { return path_ready_; });
  }

  AVFormatContext *format_ctx = nullptr;
  base::Status status = base::kStatusCodeOk;
  AVFrame *output_frame = nullptr;

  if (avformat_open_input(&format_ctx, path_.c_str(), nullptr, nullptr) < 0) {
    NNDEPLOY_LOGE("Could not open input file %s\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  do {
    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
      NNDEPLOY_LOGE("Could not find stream information\n");
      status = base::kStatusCodeErrorInvalidParam;
      break;
    }

    int video_stream_index = -1;
    for (unsigned int i = 0; i < format_ctx->nb_streams; ++i) {
      if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        video_stream_index = static_cast<int>(i);
        break;
      }
    }
    if (video_stream_index < 0) {
      NNDEPLOY_LOGE("Could not find video stream in %s\n", path_.c_str());
      status = base::kStatusCodeErrorInvalidParam;
      break;
    }

    AVCodecParameters *codecpar = format_ctx->streams[video_stream_index]->codecpar;
    const AVCodec *codec = avcodec_find_decoder(codecpar->codec_id);
    if (codec == nullptr) {
      NNDEPLOY_LOGE("Codec not found for %s\n", path_.c_str());
      status = base::kStatusCodeErrorNotSupport;
      break;
    }

    AVCodecContext *codec_ctx = avcodec_alloc_context3(codec);
    if (codec_ctx == nullptr) {
      status = base::kStatusCodeErrorOutOfMemory;
      break;
    }

    if (avcodec_parameters_to_context(codec_ctx, codecpar) < 0) {
      NNDEPLOY_LOGE("Failed to copy codec parameters to context\n");
      avcodec_free_context(&codec_ctx);
      status = base::kStatusCodeErrorInvalidParam;
      break;
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
      NNDEPLOY_LOGE("Failed to open codec\n");
      avcodec_free_context(&codec_ctx);
      status = base::kStatusCodeErrorInvalidParam;
      break;
    }

    AVPacket *packet = av_packet_alloc();
    AVFrame *decoded_frame = av_frame_alloc();
    if (packet == nullptr || decoded_frame == nullptr) {
      av_packet_free(&packet);
      av_frame_free(&decoded_frame);
      avcodec_free_context(&codec_ctx);
      status = base::kStatusCodeErrorOutOfMemory;
      break;
    }

    while (av_read_frame(format_ctx, packet) >= 0 && status == base::kStatusCodeOk) {
      if (packet->stream_index != video_stream_index) {
        av_packet_unref(packet);
        continue;
      }

      if (avcodec_send_packet(codec_ctx, packet) < 0) {
        NNDEPLOY_LOGE("Failed to send packet to decoder\n");
        status = base::kStatusCodeErrorInvalidParam;
        break;
      }

      int receive_ret = 0;
      while ((receive_ret = avcodec_receive_frame(codec_ctx, decoded_frame)) >= 0) {
        AVFrame *tmp_frame = av_frame_alloc();
        if (tmp_frame == nullptr) {
          status = base::kStatusCodeErrorOutOfMemory;
          break;
        }
        tmp_frame->format = decoded_frame->format;
        tmp_frame->width = decoded_frame->width;
        tmp_frame->height = decoded_frame->height;
        if (av_frame_get_buffer(tmp_frame, 0) < 0 ||
            av_frame_copy(tmp_frame, decoded_frame) < 0 ||
            av_frame_copy_props(tmp_frame, decoded_frame) < 0) {
          NNDEPLOY_LOGE("Failed to copy decoded frame\n");
          av_frame_free(&tmp_frame);
          status = base::kStatusCodeErrorInvalidParam;
          break;
        }
        output_frame = tmp_frame;
        width_ = decoded_frame->width;
        height_ = decoded_frame->height;
        break;
      }

      av_packet_unref(packet);
      if (output_frame != nullptr || status != base::kStatusCodeOk) {
        break;
      }
      if (receive_ret == AVERROR(EAGAIN) || receive_ret == AVERROR_EOF) {
        continue;
      } else if (receive_ret < 0) {
        NNDEPLOY_LOGE("Failed to receive frame from decoder\n");
        status = base::kStatusCodeErrorInvalidParam;
        break;
      }
    }

    av_packet_free(&packet);
    av_frame_free(&decoded_frame);
    avcodec_free_context(&codec_ctx);

    if (status != base::kStatusCodeOk) {
      break;
    }

    if (output_frame == nullptr) {
      status = base::kStatusCodeErrorInvalidParam;
    }
  } while (false);

  avformat_close_input(&format_ctx);

  if (status != base::kStatusCodeOk) {
    if (output_frame != nullptr) {
      av_frame_free(&output_frame);
    }
    return status;
  }

  outputs_[0]->set(output_frame, false);
  index_++;
  return base::kStatusCodeOk;
}

base::Status FFmpegImageEncode::init() { return base::kStatusCodeOk; }
base::Status FFmpegImageEncode::deinit() { return base::kStatusCodeOk; }

base::Status FFmpegImageEncode::setPath(const std::string &path) {
  if (path.empty()) {
    NNDEPLOY_LOGE("encode path is empty\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  path_ = path;
  path_changed_ = true;
  size_ = 1;
  this->setLoopCount(size_);
  return base::kStatusCodeOk;
}

base::Status FFmpegImageEncode::setRefPath(const std::string &ref_path) {
  ref_path_ = ref_path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status FFmpegImageEncode::run() {
  AVFrame *frame = inputs_[0]->get<AVFrame>(this);
  if (frame == nullptr) {
    NNDEPLOY_LOGE("Failed to get AVFrame from input\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  base::Status status = encodeFrameToFile(frame);
  if (status == base::kStatusCodeOk) {
    index_++;
  }
  return status;
}

base::Status FFmpegImageEncode::encodeFrameToFile(AVFrame *frame) {
  if (frame == nullptr) {
    return base::kStatusCodeErrorInvalidParam;
  }
  AVCodecID codec_id = GuessImageCodec(path_);
  if (codec_id == AV_CODEC_ID_NONE) {
    NNDEPLOY_LOGE("Unsupported image format for %s\n", path_.c_str());
    return base::kStatusCodeErrorNotSupport;
  }

  const AVCodec *codec = avcodec_find_encoder(codec_id);
  if (codec == nullptr) {
    NNDEPLOY_LOGE("Failed to find encoder for codec id %d\n", codec_id);
    return base::kStatusCodeErrorNotSupport;
  }

  AVCodecContext *codec_ctx = avcodec_alloc_context3(codec);
  if (codec_ctx == nullptr) {
    return base::kStatusCodeErrorOutOfMemory;
  }

  codec_ctx->width = frame->width;
  codec_ctx->height = frame->height;
  codec_ctx->time_base = AVRational{1, fps_ > 0 ? static_cast<int>(fps_) : 25};
  codec_ctx->pix_fmt = static_cast<AVPixelFormat>(frame->format);

  AVFrame *frame_to_encode = frame;
  AVFrame *converted_frame = nullptr;
  SwsContext *sws_ctx = nullptr;

  if (!IsPixelFormatSupported(codec, codec_ctx->pix_fmt)) {
    AVPixelFormat target_fmt = codec->pix_fmts ? codec->pix_fmts[0] : AV_PIX_FMT_YUV420P;
    sws_ctx = sws_getContext(frame->width, frame->height,
                             static_cast<AVPixelFormat>(frame->format), frame->width,
                             frame->height, target_fmt, SWS_BILINEAR, nullptr, nullptr,
                             nullptr);
    if (sws_ctx == nullptr) {
      NNDEPLOY_LOGE("Failed to create sws context for conversion\n");
      avcodec_free_context(&codec_ctx);
      return base::kStatusCodeErrorInvalidParam;
    }

    converted_frame = av_frame_alloc();
    if (converted_frame == nullptr) {
      sws_freeContext(sws_ctx);
      avcodec_free_context(&codec_ctx);
      return base::kStatusCodeErrorOutOfMemory;
    }
    converted_frame->format = target_fmt;
    converted_frame->width = frame->width;
    converted_frame->height = frame->height;
    if (av_frame_get_buffer(converted_frame, 0) < 0) {
      sws_freeContext(sws_ctx);
      av_frame_free(&converted_frame);
      avcodec_free_context(&codec_ctx);
      return base::kStatusCodeErrorInvalidParam;
    }

    if (sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                  converted_frame->data, converted_frame->linesize) <= 0) {
      sws_freeContext(sws_ctx);
      av_frame_free(&converted_frame);
      avcodec_free_context(&codec_ctx);
      return base::kStatusCodeErrorInvalidParam;
    }
    sws_freeContext(sws_ctx);
    frame_to_encode = converted_frame;
    codec_ctx->pix_fmt = target_fmt;
  }

  if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
    av_frame_free(&converted_frame);
    avcodec_free_context(&codec_ctx);
    return base::kStatusCodeErrorInvalidParam;
  }

  AVPacket *packet = av_packet_alloc();
  if (packet == nullptr) {
    av_frame_free(&converted_frame);
    avcodec_free_context(&codec_ctx);
    return base::kStatusCodeErrorOutOfMemory;
  }

  base::Status status = base::kStatusCodeOk;
  int ret = avcodec_send_frame(codec_ctx, frame_to_encode);
  if (ret < 0) {
    status = base::kStatusCodeErrorInvalidParam;
  }

  if (status == base::kStatusCodeOk) {
    ret = avcodec_receive_packet(codec_ctx, packet);
    if (ret < 0) {
      status = base::kStatusCodeErrorInvalidParam;
    }
  }

  if (status == base::kStatusCodeOk) {
    std::ofstream ofs(path_, std::ios::binary);
    if (!ofs.is_open()) {
      status = base::kStatusCodeErrorInvalidParam;
    } else {
      ofs.write(reinterpret_cast<const char *>(packet->data), packet->size);
      ofs.close();
    }
  }

  av_packet_free(&packet);
  avcodec_free_context(&codec_ctx);
  if (converted_frame != nullptr) {
    av_frame_free(&converted_frame);
  }

  return status;
}

TypeCreatelDecodeRegister g_type_create_ffmpeg_decode_node_register(
    base::kCodecTypeFFmpeg, createFFmpegDecode);
TypeCreatelDecodeSharedPtrRegister g_type_create_ffmpeg_decode_node_shared_ptr_register(
    base::kCodecTypeFFmpeg, createFFmpegDecodeSharedPtr);

TypeCreatelEncodeRegister g_type_create_ffmpeg_encode_node_register(
    base::kCodecTypeFFmpeg, createFFmpegEncode);
TypeCreatelEncodeSharedPtrRegister g_type_create_ffmpeg_encode_node_shared_ptr_register(
    base::kCodecTypeFFmpeg, createFFmpegEncodeSharedPtr);

Decode *createFFmpegDecode(base::CodecFlag flag, const std::string &name,
                           dag::Edge *output) {
  if (flag == base::kCodecFlagImage || flag == base::kCodecFlagImages) {
    return new FFmpegImageDecode(name, {}, {output}, flag);
  }
  return nullptr;
}

std::shared_ptr<Decode> createFFmpegDecodeSharedPtr(base::CodecFlag flag,
                                                    const std::string &name,
                                                    dag::Edge *output) {
  if (flag == base::kCodecFlagImage || flag == base::kCodecFlagImages) {
    return std::shared_ptr<Decode>(
        new FFmpegImageDecode(name, {}, {output}, flag));
  }
  return nullptr;
}

Encode *createFFmpegEncode(base::CodecFlag flag, const std::string &name,
                           dag::Edge *input) {
  if (flag == base::kCodecFlagImage || flag == base::kCodecFlagImages) {
    return new FFmpegImageEncode(name, {input}, {}, flag);
  }
  return nullptr;
}

std::shared_ptr<Encode> createFFmpegEncodeSharedPtr(base::CodecFlag flag,
                                                    const std::string &name,
                                                    dag::Edge *input) {
  if (flag == base::kCodecFlagImage || flag == base::kCodecFlagImages) {
    return std::shared_ptr<Encode>(
        new FFmpegImageEncode(name, {input}, {}, flag));
  }
  return nullptr;
}

REGISTER_NODE("nndeploy::codec::FFmpegImageDecode", FFmpegImageDecode);
REGISTER_NODE("nndeploy::codec::FFmpegImageEncode", FFmpegImageEncode);

}  // namespace codec
}  // namespace nndeploy
