#ifndef _NNDEPLOY_CODEC_FFMPEG_HW_CODEC_H_
#define _NNDEPLOY_CODEC_FFMPEG_HW_CODEC_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/status.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

#include <string>
#include <cstdlib>

namespace nndeploy {
namespace codec {

// Software codec configuration for architecture-optimized SW paths.
// Controls how SW encode/decode is tuned per platform.
struct NNDEPLOY_CC_API FFmpegSWCodecConfig {
  std::string codec_name;   // Explicit SW codec name override (e.g. "libx264",
                            // "h264"). Empty = auto-detect.
  std::string preset;       // Codec preset: "ultrafast","superfast","veryfast",
                            // "faster","fast","medium","slow","slower","veryslow"
  std::string tune;         // Codec tune: "film","animation","grain",
                            // "zerolatency","fastdecode","psnr","ssim"
  std::string profile;      // Codec profile: "baseline","main","high","high10",
                            // "high444"
  int crf = -1;             // Quality-based CRF (0-51). -1 = disabled, uses bitrate
  int bit_rate = 0;         // Bitrate override in bps. 0 = use default (2 Mbps)
  int threads = 0;          // Decode/encode thread count. 0 = auto (cpu_count)
  bool prefer_hw_first =    // Try HW first when hw_codec_ is set;
      true;                 // false = skip HW even if hw_codec_ is configured

  // Fill default SW codec config optimal for the current CPU architecture.
  //   x86: preset=medium, threads=auto, codec=libx264 (h264) / libx265 (hevc)
  //   ARM: preset=fast, threads=auto, codec=libx264 (h264) / hevc (built-in)
  static FFmpegSWCodecConfig getDefaultForArch();
};

// Maps FFmpegHWDeviceType to AVHWDeviceType and provides encoder/decoder names
class NNDEPLOY_CC_API FFmpegHWCodec {
 public:
  FFmpegHWCodec();
  explicit FFmpegHWCodec(base::FFmpegHWDeviceType hw_type);
  ~FFmpegHWCodec();

  // Set the HW device type
  void setHwDeviceType(base::FFmpegHWDeviceType hw_type);
  base::FFmpegHWDeviceType getHwDeviceType() const;

  // Returns true if HW acceleration is configured
  bool isEnabled() const;

  // Initialize the HW device context (call BEFORE opening codec)
  // Returns the AVHWDeviceType mapped from our enum
  AVHWDeviceType initHwDevice(base::Status *status);

  // Get the HW device context reference for assigning to codec_ctx->hw_device_ctx
  AVBufferRef *getHwDeviceRef() const;

  // Find the best matching HW decoder name for a given codec_id
  // Returns nullptr if no HW decoder available, caller to fall back to SW
  const AVCodec *findHwDecoder(AVCodecID codec_id, int width, int height);

  // Find the best matching HW encoder name for a given codec_id
  // Returns nullptr if no HW encoder available
  const AVCodec *findHwEncoder(AVCodecID codec_id, const std::string &fourcc);

  // Get the HW pixel format after decoder initialization
  AVPixelFormat getHwPixelFormat() const;

  // Set the HW pixel format (called from get_format callback)
  void setHwPixelFormat(AVPixelFormat fmt);

  // Transfer a decoded HW frame to SW memory
  // Returns the SW frame that can be converted to cv::Mat
  base::Status transferFrameToSw(AVFrame *hw_frame, AVFrame *sw_frame);

  // Upload a SW frame to HW memory (for encoding)
  base::Status transferFrameToHw(AVFrame *sw_frame, AVFrame *hw_frame);

  // Get human-readable name for this HW device
  std::string getHwDeviceName() const;

  // Cleanup HW resources
  void cleanup();

 private:
  // Map our enum to AVHWDeviceType
  AVHWDeviceType toAVHWDeviceType() const;

  // Get the decoder name suffix for a given HW type (e.g., "_qsv", "_cuvid")
  std::string getDecoderSuffix() const;

  // Get the encoder name prefix for a given HW type (e.g., "h264_nvenc", "h264_qsv")
  // Returns the full encoder codec name for a given base codec name
  std::string getEncoderName(const std::string &base_codec_name) const;

  base::FFmpegHWDeviceType hw_type_;
  AVBufferRef *hw_device_ref_;
  AVPixelFormat hw_pix_fmt_;
  bool initialized_;
};

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_FFMPEG_HW_CODEC_H_ */
