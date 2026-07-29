
#ifndef _NNDEPLOY_CODEC_GSTREAMER_WEBRTC_CODEC_H_
#define _NNDEPLOY_CODEC_GSTREAMER_WEBRTC_CODEC_H_

#include "nndeploy/codec/codec.h"

#include <thread>
#include <mutex>

#include <gst/gst.h>
#include <gst/app/app.h>
#include <gst/webrtc/webrtc.h>

namespace nndeploy {
namespace codec {

struct WebRTCContext {
  GstElement *pipeline;
  GstElement *webrtcbin;
  GstElement *appsink;
  GstElement *appsrc;
  GMainLoop *loop;
  gchar *stun_server;
  gchar *turn_server;
  gchar *turn_user;
  gchar *turn_pass;
  gboolean is_server;
  gboolean negotiation_done;
  gint video_stream_id;
  gint audio_stream_id;
  void *user_data;
};

class NNDEPLOY_CC_API GStreamerWebRtcDecode : public Decode {
 public:
  GStreamerWebRtcDecode(const std::string &name)
      : Decode(name, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::GStreamerWebRtcDecode";
    desc_ =
        "Decode WebRTC video stream using GStreamer, receives video from "
        "remote peer and outputs cv::Mat frames";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
  }
  GStreamerWebRtcDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::GStreamerWebRtcDecode";
    desc_ =
        "Decode WebRTC video stream using GStreamer, receives video from "
        "remote peer and outputs cv::Mat frames";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
  }
  virtual ~GStreamerWebRtcDecode();

  void setStunServer(const std::string &stun) { stun_server_ = stun; }
  void setTurnServer(const std::string &turn) { turn_server_ = turn; }
  void setTurnCredentials(const std::string &user, const std::string &pass) {
    turn_user_ = user;
    turn_pass_ = pass;
  }
  
  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();
  
  base::Status setRemoteSdp(const std::string &sdp);
  std::string getLocalSdp();

 private:
  WebRTCContext *ctx_ = nullptr;
  std::string stun_server_ = "stun://stun.l.google.com:19302";
  std::string turn_server_;
  std::string turn_user_;
  std::string turn_pass_;
  std::string local_sdp_;
  std::string remote_sdp_;
  GstSample *pending_sample_ = nullptr;
  std::mutex sample_mutex_;
  std::thread *event_loop_thread_ = nullptr;
  bool running_ = false;
};

class NNDEPLOY_CC_API GStreamerWebRtcEncode : public Encode {
 public:
  GStreamerWebRtcEncode(const std::string &name)
      : Encode(name, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::GStreamerWebRtcEncode";
    desc_ =
        "Encode WebRTC video stream using GStreamer, sends cv::Mat frames to "
        "remote peer";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "webrtc";
  }
  GStreamerWebRtcEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::GStreamerWebRtcEncode";
    desc_ =
        "Encode WebRTC video stream using GStreamer, sends cv::Mat frames to "
        "remote peer";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "webrtc";
  }
  virtual ~GStreamerWebRtcEncode();

  void setStunServer(const std::string &stun) { stun_server_ = stun; }
  void setTurnServer(const std::string &turn) { turn_server_ = turn; }
  void setTurnCredentials(const std::string &user, const std::string &pass) {
    turn_user_ = user;
    turn_pass_ = pass;
  }
  
  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();
  
  base::Status setRemoteSdp(const std::string &sdp);
  std::string getLocalSdp();

 private:
  WebRTCContext *ctx_ = nullptr;
  std::string stun_server_ = "stun://stun.l.google.com:19302";
  std::string turn_server_;
  std::string turn_user_;
  std::string turn_pass_;
  std::string local_sdp_;
  std::string remote_sdp_;
  std::thread *event_loop_thread_ = nullptr;
  bool running_ = false;
};

extern "C" {
void on_negotiation_needed(GstElement *webrtcbin, gpointer user_data);
void on_ice_candidate(GstElement *webrtcbin, guint mlineindex, gchar *candidate, gpointer user_data);
void on_incoming_stream(GstElement *webrtcbin, GstPad *pad, gpointer user_data);
void on_new_sample(GstAppSink *appsink, gpointer user_data);
void on_message(GstBus *bus, GstMessage *message, gpointer user_data);
}

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_GSTREAMER_WEBRTC_CODEC_H_ */
