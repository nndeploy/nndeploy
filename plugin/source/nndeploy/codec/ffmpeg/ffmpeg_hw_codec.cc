#include "nndeploy/codec/ffmpeg/ffmpeg_hw_codec.h"

#include "nndeploy/base/log.h"
#include "nndeploy/base/status.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

namespace nndeploy {
namespace codec {

// Platform detection: returns true if running on ARM (aarch64 / arm)
static inline bool isArmArch() {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || \
    defined(_M_ARM) || defined(__ARM_ARCH)
  return true;
#elif defined(__x86_64__) || defined(_M_X64) || defined(i386) || \
    defined(__i386__) || defined(__i386) || defined(_M_IX86)
  return false;
#else
  return false;
#endif
}

FFmpegSWCodecConfig FFmpegSWCodecConfig::getDefaultForArch() {
  FFmpegSWCodecConfig cfg;
  if (isArmArch()) {
    // ARM NEON platforms: lighter defaults, prefer fast over quality
    cfg.preset    = "fast";
    cfg.profile   = "high";
    cfg.threads   = 0;  // auto (cpu_count)
    cfg.crf       = 23;
    cfg.bit_rate  = 0;
  } else {
    // x86: libx264 scales well, medium is good quality/speed balance
    cfg.preset    = "medium";
    cfg.profile   = "high";
    cfg.threads   = 0;  // auto
    cfg.crf       = 23;
    cfg.bit_rate  = 0;
  }
  cfg.codec_name       = "";
  cfg.tune             = "";
  cfg.prefer_hw_first  = true;
  return cfg;
}

FFmpegHWCodec::FFmpegHWCodec()
    : hw_type_(base::kFFmpegHWDeviceNone),
      hw_device_ref_(nullptr),
      hw_pix_fmt_(AV_PIX_FMT_NONE),
      initialized_(false) {}

FFmpegHWCodec::FFmpegHWCodec(base::FFmpegHWDeviceType hw_type)
    : hw_type_(hw_type),
      hw_device_ref_(nullptr),
      hw_pix_fmt_(AV_PIX_FMT_NONE),
      initialized_(false) {}

FFmpegHWCodec::~FFmpegHWCodec() { cleanup(); }

void FFmpegHWCodec::setHwDeviceType(base::FFmpegHWDeviceType hw_type) {
  hw_type_ = hw_type;
}

base::FFmpegHWDeviceType FFmpegHWCodec::getHwDeviceType() const {
  return hw_type_;
}

bool FFmpegHWCodec::isEnabled() const {
  return hw_type_ != base::kFFmpegHWDeviceNone;
}

AVHWDeviceType FFmpegHWCodec::toAVHWDeviceType() const {
  switch (hw_type_) {
    case base::kFFmpegHWDeviceQsv:
      return AV_HWDEVICE_TYPE_QSV;
    case base::kFFmpegHWDeviceVaapi:
      return AV_HWDEVICE_TYPE_VAAPI;
    case base::kFFmpegHWDeviceCuda:
      return AV_HWDEVICE_TYPE_CUDA;
    case base::kFFmpegHWDeviceVideoToolbox:
      return AV_HWDEVICE_TYPE_VIDEOTOOLBOX;
    case base::kFFmpegHWDeviceV4l2M2m:
      return AV_HWDEVICE_TYPE_V4L2M2M;
    case base::kFFmpegHWDeviceRkmpp:
      return AV_HWDEVICE_TYPE_DRM;  // RKMPP uses DRM primitive in FFmpeg
    case base::kFFmpegHWDeviceAscend:
#ifdef AV_HWDEVICE_TYPE_ASCEND
      return AV_HWDEVICE_TYPE_ASCEND;
#else
      return AV_HWDEVICE_TYPE_NONE;
#endif
    case base::kFFmpegHWDeviceSophgo:
      // Sophgo BM uses custom FFmpeg; falls back to SW
      return AV_HWDEVICE_TYPE_NONE;
    default:
      return AV_HWDEVICE_TYPE_NONE;
  }
}

std::string FFmpegHWCodec::getDecoderSuffix() const {
  switch (hw_type_) {
    case base::kFFmpegHWDeviceQsv:
      return "_qsv";
    case base::kFFmpegHWDeviceVaapi:
      return "_vaapi";
    case base::kFFmpegHWDeviceCuda:
      return "_cuvid";
    case base::kFFmpegHWDeviceVideoToolbox:
      return "_videotoolbox";
    case base::kFFmpegHWDeviceV4l2M2m:
      return "_v4l2m2m";
    case base::kFFmpegHWDeviceRkmpp:
      return "_rkmpp";
    case base::kFFmpegHWDeviceAscend:
      return "_ascend";
    case base::kFFmpegHWDeviceSophgo:
      return "_bm";
    default:
      return "";
  }
}

std::string FFmpegHWCodec::getEncoderName(const std::string &base_codec) const {
  // HW encoder names follow specific patterns per platform
  if (base_codec == "h264") {
    switch (hw_type_) {
      case base::kFFmpegHWDeviceQsv:
        return "h264_qsv";
      case base::kFFmpegHWDeviceVaapi:
        return "h264_vaapi";
      case base::kFFmpegHWDeviceCuda:
        return "h264_nvenc";
      case base::kFFmpegHWDeviceVideoToolbox:
        return "h264_videotoolbox";
      case base::kFFmpegHWDeviceV4l2M2m:
        return "h264_v4l2m2m";
      case base::kFFmpegHWDeviceRkmpp:
        return "h264_rkmpp";
      case base::kFFmpegHWDeviceAscend:
        return "h264_ascend";
      default:
        return "";
    }
  } else if (base_codec == "hevc") {
    switch (hw_type_) {
      case base::kFFmpegHWDeviceQsv:
        return "hevc_qsv";
      case base::kFFmpegHWDeviceVaapi:
        return "hevc_vaapi";
      case base::kFFmpegHWDeviceCuda:
        return "hevc_nvenc";
      case base::kFFmpegHWDeviceVideoToolbox:
        return "hevc_videotoolbox";
      case base::kFFmpegHWDeviceV4l2M2m:
        return "hevc_v4l2m2m";
      case base::kFFmpegHWDeviceRkmpp:
        return "hevc_rkmpp";
      case base::kFFmpegHWDeviceAscend:
        return "hevc_ascend";
      default:
        return "";
    }
  } else if (base_codec == "av1" && hw_type_ == base::kFFmpegHWDeviceCuda) {
    return "av1_nvenc";
  } else if (base_codec == "mjpeg") {
    if (hw_type_ == base::kFFmpegHWDeviceQsv) {
      return "mjpeg_qsv";
    } else if (hw_type_ == base::kFFmpegHWDeviceCuda) {
      return "mjpeg_nvenc";
    }
  }
  return "";
}

AVPixelFormat FFmpegHWCodec::getHwPixelFormat() const { return hw_pix_fmt_; }

void FFmpegHWCodec::setHwPixelFormat(AVPixelFormat fmt) { hw_pix_fmt_ = fmt; }

AVHWDeviceType FFmpegHWCodec::initHwDevice(base::Status *status) {
  AVHWDeviceType av_type = toAVHWDeviceType();
  if (av_type == AV_HWDEVICE_TYPE_NONE) {
    NNDEPLOY_LOGE("Unsupported FFmpeg HW device type\n");
    if (status) *status = base::kStatusCodeErrorInvalidParam;
    return AV_HWDEVICE_TYPE_NONE;
  }

  // Free any existing device reference
  if (hw_device_ref_) {
    av_buffer_unref(&hw_device_ref_);
  }

  int ret = av_hwdevice_ctx_create(&hw_device_ref_, av_type, nullptr, nullptr,
                                   0);
  if (ret < 0) {
    NNDEPLOY_LOGE("Failed to create HW device context: %s (av_err2str not available, err=%d)\n",
                  av_type == AV_HWDEVICE_TYPE_CUDA ? "CUDA" :
                  av_type == AV_HWDEVICE_TYPE_QSV ? "QSV" :
                  av_type == AV_HWDEVICE_TYPE_VAAPI ? "VAAPI" :
                  av_type == AV_HWDEVICE_TYPE_VIDEOTOOLBOX ? "VideoToolbox" :
                  av_type == AV_HWDEVICE_TYPE_V4L2M2M ? "V4L2M2M" : "Unknown",
                  ret);
    hw_device_ref_ = nullptr;
    if (status) *status = base::kStatusCodeErrorGeneric;
    initialized_ = false;
    return AV_HWDEVICE_TYPE_NONE;
  }

  initialized_ = true;
  if (status) *status = base::kStatusCodeOk;
  return av_type;
}

AVBufferRef *FFmpegHWCodec::getHwDeviceRef() const { return hw_device_ref_; }

const AVCodec *FFmpegHWCodec::findHwDecoder(AVCodecID codec_id, int width,
                                             int height) {
  if (!isEnabled() || !initialized_) {
    return nullptr;
  }

  std::string codec_name = avcodec_get_name(codec_id);
  std::string suffix = getDecoderSuffix();
  if (suffix.empty()) {
    return nullptr;
  }

  std::string hw_name = codec_name + suffix;
  const AVCodec *hw_codec = avcodec_find_decoder_by_name(hw_name.c_str());
  if (!hw_codec) {
    NNDEPLOY_LOGI("HW decoder '%s' not found, falling back to SW\n",
                  hw_name.c_str());
    return nullptr;
  }

  // For CUDA, also need to check if the device supports this
  // (CUDA decoder names are different: h264_cuvid, hevc_cuvid, etc.)
  if (hw_type_ == base::kFFmpegHWDeviceCuda) {
    // cuvid decoders use the _cuvid suffix, which we already construct
  }

  return hw_codec;
}

const AVCodec *FFmpegHWCodec::findHwEncoder(AVCodecID codec_id,
                                              const std::string &fourcc) {
  if (!isEnabled()) {
    return nullptr;
  }

  std::string base_codec = avcodec_get_name(codec_id);
  std::string enc_name = getEncoderName(base_codec);

  if (enc_name.empty()) {
    // Fall back to fourcc-based lookup
    if (!fourcc.empty()) {
      std::string lower_fourcc = fourcc;
      for (auto &c : lower_fourcc) c = tolower(c);
      if (lower_fourcc == "avc1" || lower_fourcc == "h264") {
        enc_name = getEncoderName("h264");
      } else if (lower_fourcc == "hevc" || lower_fourcc == "h265") {
        enc_name = getEncoderName("hevc");
      }
    }
  }

  if (enc_name.empty()) {
    return nullptr;
  }

  const AVCodec *hw_codec = avcodec_find_encoder_by_name(enc_name.c_str());
  if (!hw_codec) {
    NNDEPLOY_LOGI("HW encoder '%s' not found\n", enc_name.c_str());
    return nullptr;
  }
  return hw_codec;
}

base::Status FFmpegHWCodec::transferFrameToSw(AVFrame *hw_frame,
                                                AVFrame *sw_frame) {
  if (!hw_frame || !sw_frame) {
    NNDEPLOY_LOGE("transferFrameToSw: null frame\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  sw_frame->format = AV_PIX_FMT_NONE;
  int ret = av_hwframe_transfer_data(sw_frame, hw_frame, 0);
  if (ret < 0) {
    NNDEPLOY_LOGE("Failed to transfer HW frame to SW: %d\n", ret);
    return base::kStatusCodeErrorGeneric;
  }
  return base::kStatusCodeOk;
}

base::Status FFmpegHWCodec::transferFrameToHw(AVFrame *sw_frame,
                                                AVFrame *hw_frame) {
  if (!sw_frame || !hw_frame) {
    NNDEPLOY_LOGE("transferFrameToHw: null frame\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  int ret = av_hwframe_transfer_data(hw_frame, sw_frame, 0);
  if (ret < 0) {
    NNDEPLOY_LOGE("Failed to transfer SW frame to HW: %d\n", ret);
    return base::kStatusCodeErrorGeneric;
  }
  return base::kStatusCodeOk;
}

std::string FFmpegHWCodec::getHwDeviceName() const {
  switch (hw_type_) {
    case base::kFFmpegHWDeviceQsv:
      return "Intel QSV";
    case base::kFFmpegHWDeviceVaapi:
      return "Intel VAAPI";
    case base::kFFmpegHWDeviceCuda:
      return "NVIDIA CUDA";
    case base::kFFmpegHWDeviceVideoToolbox:
      return "Apple VideoToolbox";
    case base::kFFmpegHWDeviceV4l2M2m:
      return "V4L2 M2M";
    case base::kFFmpegHWDeviceRkmpp:
      return "Rockchip MPP";
    case base::kFFmpegHWDeviceAscend:
      return "Huawei Ascend";
    case base::kFFmpegHWDeviceSophgo:
      return "Sophgo VPU";
    default:
      return "None";
  }
}

void FFmpegHWCodec::cleanup() {
  if (hw_device_ref_) {
    av_buffer_unref(&hw_device_ref_);
    hw_device_ref_ = nullptr;
  }
  hw_pix_fmt_ = AV_PIX_FMT_NONE;
  initialized_ = false;
}

}  // namespace codec
}  // namespace nndeploy
