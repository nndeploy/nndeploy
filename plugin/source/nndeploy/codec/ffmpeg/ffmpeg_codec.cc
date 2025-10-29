#include "nndeploy/codec/ffmpeg/ffmpeg_codec.h"

#include "nndeploy/base/file.h"

namespace nndeploy {
namespace codec {

namespace {
std::string ToLowerCase(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
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

base::Status FFmpegImageDecode::setupSws(SwsContext *&sws, int src_w, int src_h,
                                         AVPixelFormat src_fmt, int dst_w,
                                         int dst_h, AVPixelFormat dst_fmt) {
  if (sws) {
    sws_freeContext(sws);
  }
  sws = sws_getContext(src_w, src_h, src_fmt, dst_w, dst_h, dst_fmt,
                       SWS_BILINEAR, nullptr, nullptr, nullptr);
  if (!sws) {
    NNDEPLOY_LOGE("sws_getContext failed\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  return base::kStatusCodeOk;
}

base::Status FFmpegImageDecode::allocFrame(AVFrame *&f, int w, int h,
                                           AVPixelFormat pix_fmt) {
  f = av_frame_alloc();
  if (!f) {
    NNDEPLOY_LOGE("av_frame_alloc failed\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  f->format = pix_fmt;
  f->width = w;
  f->height = h;
  int rc = av_frame_get_buffer(f, 32);  // aligned to 32 bytes
  if (rc < 0) {
    char buf[256];
    av_strerror(rc, buf, sizeof(buf));
    NNDEPLOY_LOGE("av_frame_get_buffer failed: [%s]\n", buf);
  }
  return base::kStatusCodeOk;
}

const char *FFmpegImageDecode::getCodecName(const std::string &fmt) {
  static const std::unordered_map<std::string, const char *> codec_map = {
      {"jpeg", "mjpeg"}, {"jpg", "mjpeg"}, {"png", "png"}, {"webp", "webp"}};

  std::string fmt_lower = ToLowerCase(fmt);

  auto it = codec_map.find(fmt_lower);
  if (it != codec_map.end()) {
    return it->second;
  }
  return nullptr;
}

base::Status FFmpegImageDecode::decode(const std::string path, cv::Mat &bgr) {
  std::string ext;
  size_t pos = path.find_last_of('.');
  if (pos != std::string::npos) ext = path.substr(pos + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

  const char *cname = getCodecName(ext);
  if (!cname) {
    NNDEPLOY_LOGE("Unsupported format: %s\n", cname);
    return base::kStatusCodeErrorInvalidParam;
  }

  const AVCodec *codec = avcodec_find_decoder_by_name(cname);
  if (!codec) {
    NNDEPLOY_LOGE("Decode not found: %s\n", cname);
    return base::kStatusCodeErrorInvalidParam;
  }

  AVCodecContext *ctx = avcodec_alloc_context3(codec);
  if (!ctx) {
    NNDEPLOY_LOGE("Avcodec alloc context failed.\n");
    return base::kStatusCodeErrorNullParam;
  }

  int rc = avcodec_open2(ctx, codec, nullptr);
  if (rc < 0) {
    char buf[256];
    av_strerror(rc, buf, sizeof(buf));
    NNDEPLOY_LOGE("avcodec_open2 failed: [%s]\n", buf);
  }

  AVPacket *pkt = av_packet_alloc();
  if (!pkt) {
    NNDEPLOY_LOGE("av_packet_alloc failed.\n");
    return base::kStatusCodeErrorUnknown;
  }

  FILE *file = fopen(path.c_str(), "rb");
  if (!file) {
    NNDEPLOY_LOGE("Failed to open image: %s", path.c_str());
    return base::kStatusCodeErrorInvalidValue;
  }
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);

  uint8_t *data = (uint8_t *)malloc(size);
  fread(data, 1, size, file);
  fclose(file);

  av_new_packet(pkt, size);
  memcpy(pkt->data, data, size);

  AVFrame *frame = av_frame_alloc();
  if (!frame) {
    NNDEPLOY_LOGE("av_frame_alloc failed.\n");
    return base::kStatusCodeErrorUnknown;
  }

  if (avcodec_send_packet(ctx, pkt) < 0) {
    NNDEPLOY_LOGE("Failed to send packet to decoder.\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  if (avcodec_receive_frame(ctx, frame) < 0) {
    NNDEPLOY_LOGE("Failed to receive frame from decoder.\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  SwsContext *sws = sws_getContext(
      frame->width, frame->height, (AVPixelFormat)frame->format, frame->width,
      frame->height, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr, nullptr);
  if (!sws) {
    NNDEPLOY_LOGE("sws_getContext decode failed.\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  bgr.create(frame->height, frame->width, CV_8UC3);
  uint8_t *dst[] = {bgr.data};
  int dst_linesize[] = {(int)bgr.step};
  sws_scale(sws, frame->data, frame->linesize, 0, frame->height, dst,
            dst_linesize);

  sws_freeContext(sws);
  av_frame_free(&frame);
  av_packet_free(&pkt);
  avcodec_free_context(&ctx);

  return base::kStatusCodeOk;
}

base::Status FFmpegImageDecode::run() {
  if (index_ == 0 && parallel_type_ == base::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(path_mutex_);
    path_cv_.wait(lock, [this] { return path_ready_; });
  }
  cv::Mat *mat = new cv::Mat();
  decode(path_, *mat);
  if (mat == nullptr) {
    NNDEPLOY_LOGE("ffmpeg decode failed: path[%s].\n", path_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  outputs_[0]->set(mat, false);
  index_++;
  return base::kStatusCodeOk;
}

TypeCreatelDecodeRegister g_type_create_ffmpeg_decode_node_register(
    base::kCodecTypeFFmpeg, createFFmpegDecode);
TypeCreatelDecodeSharedPtrRegister
    g_type_create_ffmpeg_decode_node_shared_ptr_register(
        base::kCodecTypeFFmpeg, createFFmpegDecodeSharedPtr);

Decode *createFFmpegDecode(base::CodecFlag flag, const std::string &name,
                           dag::Edge *output) {
  Decode *temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = new FFmpegImageDecode(name, {}, {output}, flag);
  }
  return temp;
}

std::shared_ptr<Decode> createFFmpegDecodeSharedPtr(base::CodecFlag flag,
                                                    const std::string &name,
                                                    dag::Edge *output) {
  std::shared_ptr<Decode> temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = std::shared_ptr<FFmpegImageDecode>(
        new FFmpegImageDecode(name, {}, {output}, flag));
  }
  return temp;
}

REGISTER_NODE("nndeploy::codec::FFmpegImageDecode", FFmpegImageDecode);

}  // namespace codec
}  // namespace nndeploy