
#include "nndeploy/codec/gstreamer/gstreamer_webrtc_codec.h"

#include <gst/app/app.h>
#include <gst/pbutils/pbutils.h>

namespace nndeploy {
namespace codec {

void on_message(GstBus *bus, GstMessage *message, gpointer user_data) {
  GError *err;
  gchar *debug_info;

  switch (GST_MESSAGE_TYPE(message)) {
    case GST_MESSAGE_ERROR:
      gst_message_parse_error(message, &err, &debug_info);
      g_printerr("Error received from element %s: %s\n",
                  GST_OBJECT_NAME(message->src), err->message);
      g_printerr("Debugging information: %s\n", debug_info ? debug_info : "none");
      g_clear_error(&err);
      g_free(debug_info);
      break;
    case GST_MESSAGE_EOS:
      g_print("End-Of-Stream reached.\n");
      break;
    case GST_MESSAGE_STATE_CHANGED: {
      GstState old_state, new_state, pending_state;
      gst_message_parse_state_changed(message, &old_state, &new_state, &pending_state);
      g_print("Element %s changed state from %s to %s.\n",
              GST_OBJECT_NAME(message->src),
              gst_element_state_get_name(old_state),
              gst_element_state_get_name(new_state));
      break;
    }
    default:
      break;
  }
}

static gboolean bus_watch_callback(GstBus *bus, GstMessage *message, gpointer user_data) {
  on_message(bus, message, user_data);
  return TRUE;  // Keep watching
}

void on_ice_candidate(GstElement *webrtcbin, guint mlineindex, gchar *candidate, gpointer user_data) {
  GStreamerWebRtcDecode *decoder = reinterpret_cast<GStreamerWebRtcDecode*>(user_data);
  if (decoder) {
    NNDEPLOY_LOGI("ICE candidate: line %d, %s\n", mlineindex, candidate);
  }
  g_free(candidate);
}

void on_new_sample(GstAppSink *appsink, gpointer user_data) {
  GStreamerWebRtcDecode *decoder = reinterpret_cast<GStreamerWebRtcDecode*>(user_data);
  if (!decoder) return;

  GstSample *sample = gst_app_sink_pull_sample(appsink);
  if (!sample) return;

  std::lock_guard<std::mutex> lock(decoder->sample_mutex_);
  if (decoder->pending_sample_) {
    gst_sample_unref(decoder->pending_sample_);
  }
  decoder->pending_sample_ = sample;
}

void on_incoming_stream(GstElement *webrtcbin, GstPad *pad, gpointer user_data) {
  GStreamerWebRtcDecode *decoder = reinterpret_cast<GStreamerWebRtcDecode*>(user_data);
  if (!decoder || !decoder->ctx_) return;

  g_print("Incoming stream on pad %s\n", GST_PAD_NAME(pad));

  GstElement *depayloader = gst_element_factory_make("rtph264depay", nullptr);
  GstElement *parser = gst_element_factory_make("h264parse", nullptr);
  GstElement *decoder_elem = gst_element_factory_make("avdec_h264", nullptr);
  GstElement *convert = gst_element_factory_make("videoconvert", nullptr);
  GstElement *appsink = gst_element_factory_make("appsink", nullptr);

  if (!depayloader || !parser || !decoder_elem || !convert || !appsink) {
    g_printerr("Failed to create elements for incoming stream\n");
    return;
  }

  gst_bin_add_many(GST_BIN(decoder->ctx_->pipeline), depayloader, parser, decoder_elem, convert, appsink, nullptr);

  gst_element_link_many(depayloader, parser, decoder_elem, convert, appsink, nullptr);

  gst_element_sync_state_with_parent(depayloader);
  gst_element_sync_state_with_parent(parser);
  gst_element_sync_state_with_parent(decoder_elem);
  gst_element_sync_state_with_parent(convert);
  gst_element_sync_state_with_parent(appsink);

  GstPad *sinkpad = gst_element_get_static_pad(depayloader, "sink");
  gst_pad_link(pad, sinkpad);
  gst_object_unref(sinkpad);

  decoder->ctx_->appsink = appsink;
  
  g_object_set(appsink, "emit-signals", TRUE, "sync", FALSE, nullptr);
  g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), user_data);
}

void on_negotiation_needed(GstElement *webrtcbin, gpointer user_data) {
  GStreamerWebRtcDecode *decoder = reinterpret_cast<GStreamerWebRtcDecode*>(user_data);
  if (!decoder || !decoder->ctx_) return;

  GstWebRTCSessionDescription *offer = gst_webrtc_bin_create_offer(webrtcbin, nullptr);
  if (!offer) {
    NNDEPLOY_LOGE("Failed to create offer\n");
    return;
  }

  GstPromise *promise = gst_promise_new();
  gst_webrtc_bin_set_local_description(webrtcbin, offer, promise);

  GstStructure *s = gst_sdp_message_to_structure(offer->sdp);
  gchar *sdp_text = gst_sdp_message_as_text(offer->sdp);
  decoder->local_sdp_ = std::string(sdp_text);
  g_free(sdp_text);

  gst_webrtc_session_description_unref(offer);
  gst_promise_interrupt(promise);
  gst_promise_unref(promise);
}

GStreamerWebRtcDecode::~GStreamerWebRtcDecode() {
  deinit();
}

base::Status GStreamerWebRtcDecode::init() {
  ctx_ = new WebRTCContext();
  ctx_->pipeline = nullptr;
  ctx_->webrtcbin = nullptr;
  ctx_->appsink = nullptr;
  ctx_->loop = nullptr;
  ctx_->stun_server = g_strdup(stun_server_.c_str());
  ctx_->turn_server = turn_server_.empty() ? nullptr : g_strdup(turn_server_.c_str());
  ctx_->turn_user = turn_user_.empty() ? nullptr : g_strdup(turn_user_.c_str());
  ctx_->turn_pass = turn_pass_.empty() ? nullptr : g_strdup(turn_pass_.c_str());
  ctx_->user_data = this;
  ctx_->negotiation_done = FALSE;

  ctx_->pipeline = gst_pipeline_new("webrtc-receive-pipeline");
  
  ctx_->webrtcbin = gst_element_factory_make("webrtcbin", nullptr);
  if (!ctx_->webrtcbin) {
    NNDEPLOY_LOGE("Failed to create webrtcbin element\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  g_object_set(ctx_->webrtcbin, "stun-server", ctx_->stun_server, nullptr);
  if (ctx_->turn_server) {
    g_object_set(ctx_->webrtcbin, "turn-server", ctx_->turn_server, nullptr);
    if (ctx_->turn_user) {
      g_object_set(ctx_->webrtcbin, "turn-user", ctx_->turn_user, nullptr);
    }
    if (ctx_->turn_pass) {
      g_object_set(ctx_->webrtcbin, "turn-password", ctx_->turn_pass, nullptr);
    }
  }

  gst_bin_add(GST_BIN(ctx_->pipeline), ctx_->webrtcbin);

  g_signal_connect(ctx_->webrtcbin, "on-negotiation-needed", G_CALLBACK(on_negotiation_needed), this);
  g_signal_connect(ctx_->webrtcbin, "ice-candidate", G_CALLBACK(on_ice_candidate), this);
  g_signal_connect(ctx_->webrtcbin, "pad-added", G_CALLBACK(on_incoming_stream), this);

  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(ctx_->pipeline));
  gst_bus_add_watch(bus, reinterpret_cast<GstBusWatchFunc>(bus_watch_callback), this);
  gst_object_unref(bus);

  running_ = true;
  event_loop_thread_ = new std::thread([this]() {
    ctx_->loop = g_main_loop_new(nullptr, FALSE);
    g_main_loop_run(ctx_->loop);
  });

  gst_element_set_state(ctx_->pipeline, GST_STATE_PLAYING);

  return base::kStatusCodeOk;
}

base::Status GStreamerWebRtcDecode::deinit() {
  running_ = false;
  
  if (ctx_) {
    if (ctx_->loop) {
      g_main_loop_quit(ctx_->loop);
    }
    
    if (event_loop_thread_ && event_loop_thread_->joinable()) {
      event_loop_thread_->join();
      delete event_loop_thread_;
      event_loop_thread_ = nullptr;
    }
    
    if (ctx_->pipeline) {
      gst_element_set_state(ctx_->pipeline, GST_STATE_NULL);
      gst_object_unref(ctx_->pipeline);
    }
    
    if (pending_sample_) {
      gst_sample_unref(pending_sample_);
      pending_sample_ = nullptr;
    }
    
    g_free(ctx_->stun_server);
    g_free(ctx_->turn_server);
    g_free(ctx_->turn_user);
    g_free(ctx_->turn_pass);
    
    delete ctx_;
    ctx_ = nullptr;
  }
  
  return base::kStatusCodeOk;
}

base::Status GStreamerWebRtcDecode::setPath(const std::string &path) {
  stun_server_ = path;
  return base::kStatusCodeOk;
}

std::string GStreamerWebRtcDecode::getLocalSdp() {
  return local_sdp_;
}

base::Status GStreamerWebRtcDecode::setRemoteSdp(const std::string &sdp) {
  if (!ctx_ || !ctx_->webrtcbin) {
    return base::kStatusCodeErrorNullParam;
  }

  remote_sdp_ = sdp;
  
  GstSDPMessage *sdp_msg = nullptr;
  if (gst_sdp_message_new(&sdp_msg) != GST_SDP_OK) {
    return base::kStatusCodeErrorInvalidParam;
  }
  
  if (gst_sdp_message_parse_buffer((const guint8*)sdp.c_str(), sdp.size(), sdp_msg) != GST_SDP_OK) {
    gst_sdp_message_free(sdp_msg);
    return base::kStatusCodeErrorInvalidParam;
  }

  GstWebRTCSessionDescription *answer = gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_ANSWER, sdp_msg);
  gst_sdp_message_free(sdp_msg);

  GstPromise *promise = gst_promise_new();
  gst_webrtc_bin_set_remote_description(ctx_->webrtcbin, answer, promise);
  
  gst_webrtc_session_description_unref(answer);
  gst_promise_interrupt(promise);
  gst_promise_unref(promise);

  ctx_->negotiation_done = TRUE;
  return base::kStatusCodeOk;
}

base::Status GStreamerWebRtcDecode::run() {
  if (!ctx_ || !ctx_->webrtcbin) {
    return base::kStatusCodeErrorNullParam;
  }

  GstSample *sample = nullptr;
  {
    std::lock_guard<std::mutex> lock(sample_mutex_);
    if (pending_sample_) {
      sample = pending_sample_;
      pending_sample_ = nullptr;
    }
  }

  if (sample) {
    cv::Mat mat;
    
    GstCaps *caps = gst_sample_get_caps(sample);
    if (caps) {
      GstStructure *s = gst_caps_get_structure(caps, 0);
      gint width, height;
      if (gst_structure_get_int(s, "width", &width) && gst_structure_get_int(s, "height", &height)) {
        GstBuffer *buffer = gst_sample_get_buffer(sample);
        if (buffer) {
          GstMapInfo map;
          if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            mat = cv::Mat(height, width, CV_8UC3, map.data, map.stride);
            mat = mat.clone();
            gst_buffer_unmap(buffer, &map);
          }
        }
      }
    }
    
    cv::Mat *output_mat = new cv::Mat(mat);
    outputs_[0]->set(output_mat, false);
    
    gst_sample_unref(sample);
  }

  return base::kStatusCodeOk;
}

GStreamerWebRtcEncode::~GStreamerWebRtcEncode() {
  deinit();
}

base::Status GStreamerWebRtcEncode::init() {
  ctx_ = new WebRTCContext();
  ctx_->pipeline = nullptr;
  ctx_->webrtcbin = nullptr;
  ctx_->appsrc = nullptr;
  ctx_->appsink = nullptr;
  ctx_->loop = nullptr;
  ctx_->stun_server = g_strdup(stun_server_.c_str());
  ctx_->turn_server = turn_server_.empty() ? nullptr : g_strdup(turn_server_.c_str());
  ctx_->turn_user = turn_user_.empty() ? nullptr : g_strdup(turn_user_.c_str());
  ctx_->turn_pass = turn_pass_.empty() ? nullptr : g_strdup(turn_pass_.c_str());
  ctx_->user_data = this;
  ctx_->negotiation_done = FALSE;

  ctx_->pipeline = gst_pipeline_new("webrtc-send-pipeline");
  
  GstElement *convert = gst_element_factory_make("videoconvert", nullptr);
  GstElement *encoder = gst_element_factory_make("x264enc", nullptr);
  GstElement *payloader = gst_element_factory_make("rtph264pay", nullptr);
  ctx_->appsrc = gst_element_factory_make("appsrc", nullptr);
  ctx_->webrtcbin = gst_element_factory_make("webrtcbin", nullptr);

  if (!ctx_->appsrc || !convert || !encoder || !payloader || !ctx_->webrtcbin) {
    NNDEPLOY_LOGE("Failed to create webrtc send elements\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  g_object_set(ctx_->webrtcbin, "stun-server", ctx_->stun_server, nullptr);
  if (ctx_->turn_server) {
    g_object_set(ctx_->webrtcbin, "turn-server", ctx_->turn_server, nullptr);
    if (ctx_->turn_user) {
      g_object_set(ctx_->webrtcbin, "turn-user", ctx_->turn_user, nullptr);
    }
    if (ctx_->turn_pass) {
      g_object_set(ctx_->webrtcbin, "turn-password", ctx_->turn_pass, nullptr);
    }
  }

  // x264enc tune=0x00000004 = ultrafast, bitrate in kbps
  g_object_set(encoder, "bitrate", 2000, "tune", 0x00000004, nullptr);

  gst_bin_add_many(GST_BIN(ctx_->pipeline), ctx_->appsrc, convert, encoder, payloader, ctx_->webrtcbin, nullptr);

  if (!gst_element_link_many(ctx_->appsrc, convert, encoder, payloader, nullptr)) {
    NNDEPLOY_LOGE("Failed to link encode pipeline elements\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  GstPad *srcpad = gst_element_get_static_pad(payloader, "src");
  if (!srcpad) {
    NNDEPLOY_LOGE("Failed to get src pad from payloader\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  GstPad *sinkpad = gst_element_get_request_pad(ctx_->webrtcbin, "sink_%u");
  if (!sinkpad) {
    NNDEPLOY_LOGE("Failed to get request pad from webrtcbin\n");
    gst_object_unref(srcpad);
    return base::kStatusCodeErrorInvalidParam;
  }
  GstPadLinkReturn link_ret = gst_pad_link(srcpad, sinkpad);
  gst_object_unref(srcpad);
  gst_object_unref(sinkpad);
  if (link_ret != GST_PAD_LINK_OK) {
    NNDEPLOY_LOGE("Failed to link payloader to webrtcbin\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(ctx_->pipeline));
  gst_bus_add_watch(bus, reinterpret_cast<GstBusWatchFunc>(bus_watch_callback), this);
  gst_object_unref(bus);

  running_ = true;
  event_loop_thread_ = new std::thread([this]() {
    ctx_->loop = g_main_loop_new(nullptr, FALSE);
    g_main_loop_run(ctx_->loop);
  });

  gst_element_set_state(ctx_->pipeline, GST_STATE_PLAYING);

  return base::kStatusCodeOk;
}

base::Status GStreamerWebRtcEncode::deinit() {
  running_ = false;
  
  if (ctx_) {
    if (ctx_->loop) {
      g_main_loop_quit(ctx_->loop);
    }
    
    if (event_loop_thread_ && event_loop_thread_->joinable()) {
      event_loop_thread_->join();
      delete event_loop_thread_;
      event_loop_thread_ = nullptr;
    }
    
    if (ctx_->pipeline) {
      gst_element_set_state(ctx_->pipeline, GST_STATE_NULL);
      gst_object_unref(ctx_->pipeline);
    }
    
    g_free(ctx_->stun_server);
    g_free(ctx_->turn_server);
    g_free(ctx_->turn_user);
    g_free(ctx_->turn_pass);
    
    delete ctx_;
    ctx_ = nullptr;
  }
  
  return base::kStatusCodeOk;
}

base::Status GStreamerWebRtcEncode::setPath(const std::string &path) {
  stun_server_ = path;
  return base::kStatusCodeOk;
}

std::string GStreamerWebRtcEncode::getLocalSdp() {
  return local_sdp_;
}

base::Status GStreamerWebRtcEncode::setRemoteSdp(const std::string &sdp) {
  if (!ctx_ || !ctx_->webrtcbin) {
    return base::kStatusCodeErrorNullParam;
  }

  remote_sdp_ = sdp;
  
  GstSDPMessage *sdp_msg = nullptr;
  if (gst_sdp_message_new(&sdp_msg) != GST_SDP_OK) {
    return base::kStatusCodeErrorInvalidParam;
  }
  
  if (gst_sdp_message_parse_buffer((const guint8*)sdp.c_str(), sdp.size(), sdp_msg) != GST_SDP_OK) {
    gst_sdp_message_free(sdp_msg);
    return base::kStatusCodeErrorInvalidParam;
  }

  GstWebRTCSessionDescription *offer = gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_OFFER, sdp_msg);
  gst_sdp_message_free(sdp_msg);

  GstPromise *promise = gst_promise_new();
  gst_webrtc_bin_set_remote_description(ctx_->webrtcbin, offer, promise);
  
  GstWebRTCSessionDescription *answer = gst_webrtc_bin_create_answer(ctx_->webrtcbin, nullptr);
  if (answer) {
    gchar *sdp_text = gst_sdp_message_as_text(answer->sdp);
    local_sdp_ = std::string(sdp_text);
    g_free(sdp_text);
    
    GstPromise *answer_promise = gst_promise_new();
    gst_webrtc_bin_set_local_description(ctx_->webrtcbin, answer, answer_promise);
    gst_promise_interrupt(answer_promise);
    gst_promise_unref(answer_promise);
    
    gst_webrtc_session_description_unref(answer);
  }
  
  gst_webrtc_session_description_unref(offer);
  gst_promise_interrupt(promise);
  gst_promise_unref(promise);

  ctx_->negotiation_done = TRUE;
  return base::kStatusCodeOk;
}

base::Status GStreamerWebRtcEncode::run() {
  if (!ctx_ || !ctx_->appsrc) {
    return base::kStatusCodeErrorNullParam;
  }

  cv::Mat *mat = inputs_[0]->getCvMat(this);
  if (!mat || mat->empty()) {
    return base::kStatusCodeOk;
  }

  GstBuffer *buffer = gst_buffer_new_and_alloc(mat->rows * mat->step[0]);
  GstMapInfo map;
  gst_buffer_map(buffer, &map, GST_MAP_WRITE);
  memcpy(map.data, mat->data, mat->rows * mat->step[0]);
  gst_buffer_unmap(buffer, &map);

  GstCaps *caps = gst_caps_new_simple("video/x-raw",
                                       "format", G_TYPE_STRING, "BGR",
                                       "width", G_TYPE_INT, mat->cols,
                                       "height", G_TYPE_INT, mat->rows,
                                       "framerate", GST_TYPE_FRACTION, 30, 1,
                                       nullptr);
  
  gst_app_src_set_caps(GST_APP_SRC(ctx_->appsrc), caps);
  gst_caps_unref(caps);

  gst_app_src_push_buffer(GST_APP_SRC(ctx_->appsrc), buffer);

  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::codec::GStreamerWebRtcDecode", GStreamerWebRtcDecode);
REGISTER_NODE("nndeploy::codec::GStreamerWebRtcEncode", GStreamerWebRtcEncode);

}  // namespace codec
}  // namespace nndeploy
