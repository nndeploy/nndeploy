
#include "nndeploy/codec/gstreamer/gstreamer_codec.h"

#include "nndeploy/base/file.h"

namespace nndeploy {
namespace codec {

// Helper: Convert GstSample to cv::Mat (BGR format)
static cv::Mat gstSampleToMat(GstSample *sample) {
  cv::Mat mat;
  if (!sample) {
    return mat;
  }

  GstCaps *caps = gst_sample_get_caps(sample);
  if (!caps) {
    return mat;
  }

  GstStructure *s = gst_caps_get_structure(caps, 0);
  if (!s) {
    return mat;
  }

  gint width, height;
  if (!gst_structure_get_int(s, "width", &width) ||
      !gst_structure_get_int(s, "height", &height)) {
    return mat;
  }

  GstBuffer *buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    return mat;
  }

  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    return mat;
  }

  // Determine format from caps
  const gchar *format_str = gst_structure_get_string(s, "format");
  if (format_str && (strcmp(format_str, "BGR") == 0 || strcmp(format_str, "RGB") == 0)) {
    mat = cv::Mat(height, width, CV_8UC3, map.data, map.stride);
  } else if (format_str && strcmp(format_str, "GRAY8") == 0) {
    mat = cv::Mat(height, width, CV_8UC1, map.data, map.stride);
  } else if (map.size == (size_t)(width * height * 3)) {
    // Assume BGR
    mat = cv::Mat(height, width, CV_8UC3, map.data, width * 3);
  } else if (map.size == (size_t)(width * height * 4)) {
    // RGBA or BGRA
    mat = cv::Mat(height, width, CV_8UC4, map.data, width * 4);
  } else if (map.size == (size_t)(width * height)) {
    // GRAY8
    mat = cv::Mat(height, width, CV_8UC1, map.data, width);
  } else {
    mat = cv::Mat(height, width, CV_8UC3, map.data, map.stride);
  }

  // Make a copy since the buffer may be freed
  if (!mat.empty()) {
    mat = mat.clone();
  }

  gst_buffer_unmap(buffer, &map);
  return mat;
}

// Helper: Build decode pipeline string for files
static std::string buildDecodePipeline(const std::string &path,
                                        const std::string &element) {
  // For images: filesrc -> decodebin -> videoconvert -> appsink
  // For video:  filesrc -> decodebin -> videoconvert -> appsink
  std::string pipeline_str = "filesrc location=" + path + " ! ";
  pipeline_str += "decodebin ! ";
  pipeline_str += "videoconvert ! ";
  pipeline_str += "video/x-raw,format=BGR ! ";
  pipeline_str += element;
  return pipeline_str;
}

// Helper: Build camera decode pipeline string
static std::string buildCameraDecodePipeline(const std::string &device,
                                             const std::string &element) {
  std::string source = device.empty() ? "v4l2src device=/dev/video0" :
                       (device.find("/dev/") == 0 ? "v4l2src device=" + device :
                                                   "v4l2src device=/dev/video" + device);
  std::string pipeline_str = source + " ! ";
  pipeline_str += "videoconvert ! ";
  pipeline_str += "video/x-raw,format=BGR ! ";
  pipeline_str += element;
  return pipeline_str;
}

// ==================== GStreamerImageDecode ====================

base::Status GStreamerImageDecode::init() { return base::kStatusCodeOk; }

base::Status GStreamerImageDecode::deinit() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  if (source_) { gst_object_unref(source_); }
  if (decode_) { gst_object_unref(decode_); }
  if (convert_) { gst_object_unref(convert_); }
  if (sink_) { gst_object_unref(sink_); }
  if (sample_) { gst_sample_unref(sample_); }
  source_ = nullptr;
  decode_ = nullptr;
  convert_ = nullptr;
  sink_ = nullptr;
  sample_ = nullptr;
  return base::kStatusCodeOk;
}

base::Status GStreamerImageDecode::setPath(const std::string &path) {
  if (!base::exists(path)) {
    NNDEPLOY_LOGE("path[%s] is not exists!\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  deinit();
  path_ = path;

  std::string pipeline_str = buildDecodePipeline(path, "appsink name=sink");
  GError *error = nullptr;
  pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
  if (error) {
    NNDEPLOY_LOGE("Failed to create pipeline: %s\n", error->message);
    g_error_free(error);
    return base::kStatusCodeErrorInvalidParam;
  }

  sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (!sink_) {
    NNDEPLOY_LOGE("Cannot find appsink\n");
    deinit();
    return base::kStatusCodeErrorInvalidParam;
  }

  gst_app_sink_set_emit_signals(GST_APP_SINK(sink_), FALSE);

  GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PAUSED);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    NNDEPLOY_LOGE("Failed to set pipeline to PAUSED\n");
    deinit();
    return base::kStatusCodeErrorInvalidParam;
  }

  // Get video info
  GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
  if (sample) {
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *s = gst_caps_get_structure(caps, 0);
    gst_structure_get_int(s, "width", &width_);
    gst_structure_get_int(s, "height", &height_);
    last_mat_ = gstSampleToMat(sample);
    gst_sample_unref(sample);
  }

  size_ = 1;
  loop_count_ = size_;

  return base::kStatusCodeOk;
}

base::Status GStreamerImageDecode::run() {
  if (!pipeline_ || !sink_) {
    return base::kStatusCodeErrorNullParam;
  }

  if (index_ >= size_) {
    return base::kStatusCodeOk;
  }

  if (gst_app_sink_is_eos(GST_APP_SINK(sink_))) {
    index_++;
    return base::kStatusCodeOk;
  }

  GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
  if (!sample) {
    return base::kStatusCodeErrorInvalidParam;
  }

  cv::Mat *mat = new cv::Mat(gstSampleToMat(sample));
  gst_sample_unref(sample);

  outputs_[0]->set(mat, false);
  index_++;
  return base::kStatusCodeOk;
}

// ==================== GStreamerImagesDecode ====================

base::Status GStreamerImagesDecode::init() { return base::kStatusCodeOk; }

base::Status GStreamerImagesDecode::deinit() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  if (sink_) { gst_object_unref(sink_); }
  if (sample_) { gst_sample_unref(sample_); }
  sink_ = nullptr;
  sample_ = nullptr;
  images_.clear();
  return base::kStatusCodeOk;
}

base::Status GStreamerImagesDecode::setPath(const std::string &path) {
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

  // Open first image to get dimensions
  if (!images_.empty()) {
    std::string first_pipeline_str = buildDecodePipeline(images_[0], "appsink name=sink");
    GError *error = nullptr;
    pipeline_ = gst_parse_launch(first_pipeline_str.c_str(), &error);
    if (!error) {
      sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
      if (sink_) {
        gst_element_set_state(pipeline_, GST_STATE_PAUSED);
        GstSample *s = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
        if (s) {
          GstCaps *caps = gst_sample_get_caps(s);
          GstStructure *st = gst_caps_get_structure(caps, 0);
          gst_structure_get_int(st, "width", &width_);
          gst_structure_get_int(st, "height", &height_);
          gst_sample_unref(s);
        }
      }
    }
  }

  return base::kStatusCodeOk;
}

base::Status GStreamerImagesDecode::run() {
  if (index_ >= size_) {
    return base::kStatusCodeOk;
  }

  // Open pipeline for current image
  deinit();
  std::string current_path = images_[index_];
  std::string pipeline_str = buildDecodePipeline(current_path, "appsink name=sink");
  GError *error = nullptr;
  pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
  if (error) {
    NNDEPLOY_LOGE("Failed to create pipeline for %s: %s\n",
                  current_path.c_str(), error->message);
    g_error_free(error);
    return base::kStatusCodeErrorInvalidParam;
  }

  sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (sink_) {
    gst_element_set_state(pipeline_, GST_STATE_PAUSED);
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
    if (sample) {
      cv::Mat *mat = new cv::Mat(gstSampleToMat(sample));
      outputs_[0]->set(mat, false);
      gst_sample_unref(sample);
    }
  }

  index_++;
  return base::kStatusCodeOk;
}

// ==================== GStreamerVideoDecode ====================

base::Status GStreamerVideoDecode::init() { return base::kStatusCodeOk; }

base::Status GStreamerVideoDecode::deinit() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  if (bus_) {
    gst_object_unref(bus_);
    bus_ = nullptr;
  }
  if (sink_) { gst_object_unref(sink_); }
  if (sample_) { gst_sample_unref(sample_); }
  sink_ = nullptr;
  sample_ = nullptr;
  return base::kStatusCodeOk;
}

base::Status GStreamerVideoDecode::setPath(const std::string &path) {
  if (!base::exists(path)) {
    NNDEPLOY_LOGE("path[%s] is not exists!\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  deinit();
  index_ = 0;
  path_ = path;

  std::string pipeline_str = buildDecodePipeline(path, "appsink name=sink emit-signals=FALSE");
  GError *error = nullptr;
  pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
  if (error) {
    NNDEPLOY_LOGE("Failed to create pipeline: %s\n", error->message);
    g_error_free(error);
    return base::kStatusCodeErrorInvalidParam;
  }

  bus_ = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
  sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (!sink_) {
    NNDEPLOY_LOGE("Cannot find appsink\n");
    deinit();
    return base::kStatusCodeErrorInvalidParam;
  }

  GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    NNDEPLOY_LOGE("Failed to set pipeline to PLAYING\n");
    deinit();
    return base::kStatusCodeErrorInvalidParam;
  }

  // Wait for pipeline to preroll
  GstMessage *msg = gst_bus_timed_pop_filtered(bus_, GST_CLOCK_TIME_NONE,
                                                (GstMessageType)(GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_ERROR));
  if (msg) {
    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
      GError *err;
      gchar *debug;
      gst_message_parse_error(msg, &err, &debug);
      NNDEPLOY_LOGE("Pipeline error: %s\n", err->message);
      g_error_free(err);
      g_free(debug);
    }
    gst_message_unref(msg);
  }

  // Get video info from first frame
  GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
  if (sample) {
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *s = gst_caps_get_structure(caps, 0);
    gst_structure_get_int(s, "width", &width_);
    gst_structure_get_int(s, "height", &height_);
    gst_sample_unref(sample);

    // Estimate size from duration
    GstFormat fmt = GST_FORMAT_TIME;
    if (gst_element_query_duration(pipeline_, fmt, nullptr)) {
      gint64 duration;
      gst_element_query_duration(pipeline_, fmt, &duration);
      if (duration > 0) {
        fps_ = 30.0;  // Default fps
        size_ = (int)(duration / 1000000000.0 * fps_);
        if (size_ <= 0) size_ = INT_MAX;
      }
    }
  }

  if (size_ <= 0) size_ = INT_MAX;
  loop_count_ = size_;

  return base::kStatusCodeOk;
}

base::Status GStreamerVideoDecode::run() {
  if (!pipeline_ || !sink_) {
    return base::kStatusCodeErrorNullParam;
  }

  if (index_ >= size_ && size_ > 0) {
    return base::kStatusCodeOk;
  }

  if (gst_app_sink_is_eos(GST_APP_SINK(sink_))) {
    if (index_ == 0) {
      NNDEPLOY_LOGW("Video file has no frames\n");
    }
    index_ = size_;  // Signal end
    return base::kStatusCodeOk;
  }

  GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
  if (!sample) {
    return base::kStatusCodeOk;  // No frame available yet
  }

  cv::Mat *mat = new cv::Mat(gstSampleToMat(sample));
  gst_sample_unref(sample);

  outputs_[0]->set(mat, false);
  index_++;
  return base::kStatusCodeOk;
}

// ==================== GStreamerCameraDecode ====================

base::Status GStreamerCameraDecode::init() {
  size_ = 0;
  return base::kStatusCodeOk;
}

base::Status GStreamerCameraDecode::deinit() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  if (bus_) {
    gst_object_unref(bus_);
    bus_ = nullptr;
  }
  if (sink_) { gst_object_unref(sink_); }
  if (sample_) { gst_sample_unref(sample_); }
  sink_ = nullptr;
  sample_ = nullptr;
  return base::kStatusCodeOk;
}

base::Status GStreamerCameraDecode::setPath(const std::string &path) {
  deinit();
  index_ = 0;
  path_ = path;

  std::string pipeline_str = buildCameraDecodePipeline(path, "appsink name=sink emit-signals=FALSE");
  GError *error = nullptr;
  pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
  if (error) {
    NNDEPLOY_LOGE("Failed to create camera pipeline: %s\n", error->message);
    g_error_free(error);
    return base::kStatusCodeErrorInvalidParam;
  }

  bus_ = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
  sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (!sink_) {
    NNDEPLOY_LOGE("Cannot find appsink\n");
    deinit();
    return base::kStatusCodeErrorInvalidParam;
  }

  GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    NNDEPLOY_LOGE("Failed to set camera pipeline to PLAYING\n");
    deinit();
    return base::kStatusCodeErrorInvalidParam;
  }

  // Get camera info from first frame
  GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
  if (sample) {
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *s = gst_caps_get_structure(caps, 0);
    gst_structure_get_int(s, "width", &width_);
    gst_structure_get_int(s, "height", &height_);
    gst_sample_unref(sample);
  }

  size_ = INT_MAX;
  loop_count_ = size_;

  return base::kStatusCodeOk;
}

base::Status GStreamerCameraDecode::run() {
  if (!pipeline_ || !sink_) {
    return base::kStatusCodeErrorNullParam;
  }

  GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
  if (!sample) {
    return base::kStatusCodeOk;  // No frame available yet
  }

  cv::Mat *mat = new cv::Mat(gstSampleToMat(sample));
  gst_sample_unref(sample);

  outputs_[0]->set(mat, false);
  index_++;
  return base::kStatusCodeOk;
}

// ==================== GStreamerImageEncode ====================

base::Status GStreamerImageEncode::init() {
  eos_ = false;
  pending_buffer_ = nullptr;
  return base::kStatusCodeOk;
}

base::Status GStreamerImageEncode::deinit() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  if (source_) { gst_object_unref(source_); }
  if (convert_) { gst_object_unref(convert_); }
  if (encode_) { gst_object_unref(encode_); }
  if (sink_) { gst_object_unref(sink_); }
  source_ = nullptr;
  convert_ = nullptr;
  encode_ = nullptr;
  sink_ = nullptr;
  if (pending_buffer_) {
    gst_buffer_unref(pending_buffer_);
    pending_buffer_ = nullptr;
  }
  eos_ = false;
  return base::kStatusCodeOk;
}

base::Status GStreamerImageEncode::setRefPath(const std::string &ref_path) {
  ref_path_ = ref_path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status GStreamerImageEncode::setPath(const std::string &path) {
  path_ = path;
  path_changed_ = true;
  size_ = 1;
  return base::kStatusCodeOk;
}

base::Status GStreamerImageEncode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  if (!mat || mat->empty()) {
    NNDEPLOY_LOGE("Input mat is empty\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  if (!pipeline_ || path_changed_) {
    deinit();

    // Build image encoding pipeline: appsrc ! videoconvert ! jpegenc ! filesink
    std::string enc_elem = path_.find(".png") != std::string::npos ? "pngenc" : "jpegenc";
    std::string pipeline_str = "appsrc name=src ! ";
    pipeline_str += "videoconvert ! ";
    pipeline_str += enc_elem + " ! ";
    pipeline_str += "filesink location=" + path_;

    GError *error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) {
      NNDEPLOY_LOGE("Failed to create encode pipeline: %s\n", error->message);
      g_error_free(error);
      return base::kStatusCodeErrorInvalidParam;
    }

    source_ = gst_bin_get_by_name(GST_BIN(pipeline_), "src");
    sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "filesink0");

    if (!source_) {
      NNDEPLOY_LOGE("Cannot find appsrc\n");
      deinit();
      return base::kStatusCodeErrorInvalidParam;
    }

    // Set caps for appsrc
    GstCaps *caps = gst_caps_new_simple("video/x-raw",
                                         "format", G_TYPE_STRING, "BGR",
                                         "width", G_TYPE_INT, mat->cols,
                                         "height", G_TYPE_INT, mat->rows,
                                         "framerate", GST_TYPE_FRACTION, 25, 1,
                                         nullptr);
    gst_app_src_set_caps(GST_APP_SRC(source_), caps);
    gst_caps_unref(caps);

    // Set to PAUSED first
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PAUSED);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      NNDEPLOY_LOGE("Failed to set encode pipeline to PAUSED\n");
      deinit();
      return base::kStatusCodeErrorInvalidParam;
    }

    path_changed_ = false;
  }

  // Create buffer from cv::Mat (BGR)
  GstBuffer *buffer = gst_buffer_new_and_alloc(mat->rows * mat->step[0]);
  GstMapInfo map;
  gst_buffer_map(buffer, &map, GST_MAP_WRITE);
  memcpy(map.data, mat->data, mat->rows * mat->step[0]);
  gst_buffer_unmap(buffer, &map);

  // Set timestamp
  GST_BUFFER_PTS(buffer) = 0;
  GST_BUFFER_DURATION(buffer) = GST_SECOND / 25;

  GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(source_), buffer);
  if (ret != GST_FLOW_OK) {
    NNDEPLOY_LOGE("Failed to push buffer\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // Send end-of-stream
  gst_app_src_end_of_stream(GST_APP_SRC(source_));

  // Start the pipeline
  gst_element_set_state(pipeline_, GST_STATE_PLAYING);

  // Wait for completion
  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
  GstMessage *msg = gst_bus_timed_pop_filtered(bus, 5 * GST_SECOND,
                                                (GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  if (msg) {
    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
      GError *err;
      gchar *debug;
      gst_message_parse_error(msg, &err, &debug);
      NNDEPLOY_LOGE("Encode pipeline error: %s\n", err->message);
      g_error_free(err);
      g_free(debug);
    }
    gst_message_unref(msg);
  }
  gst_object_unref(bus);

  return base::kStatusCodeOk;
}

// ==================== GStreamerImagesEncode ====================

base::Status GStreamerImagesEncode::init() {
  index_gs_ = 0;
  return base::kStatusCodeOk;
}

base::Status GStreamerImagesEncode::deinit() {
  return base::kStatusCodeOk;
}

base::Status GStreamerImagesEncode::setRefPath(const std::string &ref_path) {
  ref_path_ = ref_path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status GStreamerImagesEncode::setPath(const std::string &path) {
  path_ = path;
  index_gs_ = 0;
  if (!base::isDirectory(path)) {
    NNDEPLOY_LOGE("path[%s] is not Directory!\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status GStreamerImagesEncode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  if (!mat || mat->empty()) {
    NNDEPLOY_LOGE("Input mat is empty\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  std::string full_path = base::joinPath(path_, std::to_string(index_gs_) + ".jpg");

  // Create pipeline for single image
  std::string pipeline_str = "appsrc ! videoconvert ! jpegenc ! filesink location=" + full_path;
  GError *error = nullptr;
  GstElement *pipe = gst_parse_launch(pipeline_str.c_str(), &error);
  if (error) {
    NNDEPLOY_LOGE("Failed to create encode pipeline: %s\n", error->message);
    g_error_free(error);
    return base::kStatusCodeErrorInvalidParam;
  }

  GstElement *src = gst_bin_get_by_name(GST_BIN(pipe), "src");
  if (src) {
    GstCaps *caps = gst_caps_new_simple("video/x-raw",
                                         "format", G_TYPE_STRING, "BGR",
                                         "width", G_TYPE_INT, mat->cols,
                                         "height", G_TYPE_INT, mat->rows,
                                         "framerate", GST_TYPE_FRACTION, 25, 1,
                                         nullptr);
    gst_app_src_set_caps(GST_APP_SRC(src), caps);
    gst_caps_unref(caps);

    gst_element_set_state(pipe, GST_STATE_PLAYING);

    GstBuffer *buffer = gst_buffer_new_and_alloc(mat->rows * mat->step[0]);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, mat->data, mat->rows * mat->step[0]);
    gst_buffer_unmap(buffer, &map);
    GST_BUFFER_PTS(buffer) = 0;
    GST_BUFFER_DURATION(buffer) = GST_SECOND / 25;

    gst_app_src_push_buffer(GST_APP_SRC(src), buffer);
    gst_app_src_end_of_stream(GST_APP_SRC(src));

    GstBus *bus = gst_element_get_bus(pipe);
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, 5 * GST_SECOND,
                                                  (GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    if (msg) gst_message_unref(msg);
    gst_object_unref(bus);
    gst_object_unref(src);
  }

  gst_element_set_state(pipe, GST_STATE_NULL);
  gst_object_unref(pipe);

  index_gs_++;
  return base::kStatusCodeOk;
}

// ==================== GStreamerVideoEncode ====================

base::Status GStreamerVideoEncode::init() {
  eos_ = false;
  pending_buffer_ = nullptr;
  return base::kStatusCodeOk;
}

base::Status GStreamerVideoEncode::deinit() {
  if (pipeline_) {
    // Send EOS and wait
    if (!eos_ && source_) {
      gst_app_src_end_of_stream(GST_APP_SRC(source_));
    }
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, 5 * GST_SECOND,
                                                  (GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    if (msg) gst_message_unref(msg);
    gst_object_unref(bus);

    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  if (source_) { gst_object_unref(source_); }
  if (convert_) { gst_object_unref(convert_); }
  if (encode_) { gst_object_unref(encode_); }
  if (mux_) { gst_object_unref(mux_); }
  if (sink_) { gst_object_unref(sink_); }
  source_ = nullptr;
  convert_ = nullptr;
  encode_ = nullptr;
  mux_ = nullptr;
  sink_ = nullptr;
  eos_ = false;
  return base::kStatusCodeOk;
}

base::Status GStreamerVideoEncode::setRefPath(const std::string &ref_path) {
  ref_path_ = ref_path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status GStreamerVideoEncode::setPath(const std::string &path) {
  path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status GStreamerVideoEncode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  if (!mat || mat->empty()) {
    NNDEPLOY_LOGE("Input mat is empty\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  if (!pipeline_ || path_changed_) {
    deinit();

    if (fps_ <= 0) fps_ = 25.0;

    // Build video encoding pipeline with muxer
    std::string enc_codec = fourcc_ == "h265" ? "video/x-h265" : "video/x-h264";
    std::string mux_format = path_.find(".mkv") != std::string::npos ? "matroskamux" : "mp4mux";

    std::string pipeline_str = "appsrc name=src ! ";
    pipeline_str += "videoconvert ! ";
    pipeline_str += "videoscale ! ";
    pipeline_str += "video/x-raw,format=I420,width=" + std::to_string(width_) + ",height=" + std::to_string(height_) + " ! ";
    pipeline_str += "x264enc bitrate=2000 tune=zerolatency ! ";
    pipeline_str += mux_format + " name=mux ! ";
    pipeline_str += "filesink location=" + path_;

    GError *error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) {
      NNDEPLOY_LOGE("Failed to create video encode pipeline: %s\n", error->message);
      g_error_free(error);
      return base::kStatusCodeErrorInvalidParam;
    }

    source_ = gst_bin_get_by_name(GST_BIN(pipeline_), "src");
    if (!source_) {
      NNDEPLOY_LOGE("Cannot find appsrc\n");
      deinit();
      return base::kStatusCodeErrorInvalidParam;
    }

    GstCaps *caps = gst_caps_new_simple("video/x-raw",
                                         "format", G_TYPE_STRING, "BGR",
                                         "width", G_TYPE_INT, mat->cols,
                                         "height", G_TYPE_INT, mat->rows,
                                         "framerate", GST_TYPE_FRACTION, (int)fps_, 1,
                                         nullptr);
    gst_app_src_set_caps(GST_APP_SRC(source_), caps);
    gst_caps_unref(caps);
    gst_app_src_set_stream_type(GST_APP_SRC(source_), GST_APP_STREAM_TYPE_STREAM);

    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      NNDEPLOY_LOGE("Failed to set video encode pipeline to PLAYING\n");
      deinit();
      return base::kStatusCodeErrorInvalidParam;
    }

    path_changed_ = false;
  }

  // Create buffer from cv::Mat (BGR)
  GstBuffer *buffer = gst_buffer_new_and_alloc(mat->rows * mat->step[0]);
  GstMapInfo map;
  gst_buffer_map(buffer, &map, GST_MAP_WRITE);
  memcpy(map.data, mat->data, mat->rows * mat->step[0]);
  gst_buffer_unmap(buffer, &map);

  // Set timestamp based on frame index
  GST_BUFFER_PTS(buffer) = index_ * (GST_SECOND / (guint64)fps_);
  GST_BUFFER_DURATION(buffer) = GST_SECOND / (guint64)fps_;

  GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(source_), buffer);
  if (ret != GST_FLOW_OK) {
    NNDEPLOY_LOGE("Failed to push video frame buffer\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  index_++;
  return base::kStatusCodeOk;
}

// ==================== GStreamerCameraEncode ====================

base::Status GStreamerCameraEncode::init() {
  eos_ = false;
  pending_buffer_ = nullptr;
  return base::kStatusCodeOk;
}

base::Status GStreamerCameraEncode::deinit() {
  return GStreamerVideoEncode::deinit();
}

base::Status GStreamerCameraEncode::setRefPath(const std::string &ref_path) {
  ref_path_ = ref_path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status GStreamerCameraEncode::setPath(const std::string &path) {
  path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status GStreamerCameraEncode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  if (!mat || mat->empty()) {
    NNDEPLOY_LOGE("Input mat is empty\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  // Camera encode uses the same logic as video encode
  return GStreamerVideoEncode::run();
}

// ==================== GStreamerStreamDecode ====================

base::Status GStreamerStreamDecode::init() { return base::kStatusCodeOk; }

base::Status GStreamerStreamDecode::deinit() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  if (sink_) {
    gst_object_unref(sink_);
    sink_ = nullptr;
  }
  if (bus_) {
    gst_object_unref(bus_);
    bus_ = nullptr;
  }
  if (sample_) {
    gst_sample_unref(sample_);
    sample_ = nullptr;
  }
  width_gs_ = 0;
  height_gs_ = 0;
  return base::kStatusCodeOk;
}

static std::string buildStreamDecodePipeline(const std::string &url) {
  // Build a robust GStreamer pipeline for network streaming.
  // Uses decodebin for automatic codec detection (H.264, H.265, etc.)
  // rtspsrc uses TCP transport for reliability.
  std::string pipeline_str;
  if (url.find("rtsp://") == 0 || url.find("rtsps://") == 0) {
    pipeline_str = "rtspsrc location=\"" + url + "\" latency=0 buffer-mode=0 protocols=tcp ! ";
    pipeline_str += "decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=FALSE";
  } else if (url.find("http://") == 0 || url.find("https://") == 0) {
    pipeline_str = "souphttpsrc location=\"" + url + "\" ! ";
    pipeline_str += "decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=FALSE";
  } else {
    // Fallback: try decodebin for any stream type
    pipeline_str = "uridecodebin uri=\"" + url + "\" ! ";
    pipeline_str += "videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=FALSE";
  }
  return pipeline_str;
}

base::Status GStreamerStreamDecode::setPath(const std::string &path) {
  deinit();
  index_ = 0;
  path_ = path;

  std::string pipeline_str = buildStreamDecodePipeline(path);
  NNDEPLOY_LOGI("GStreamer stream pipeline: %s\n", pipeline_str.c_str());

  GError *error = nullptr;
  pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
  if (error) {
    NNDEPLOY_LOGE("Failed to create stream pipeline: %s\n", error->message);
    g_error_free(error);
    return base::kStatusCodeErrorInvalidParam;
  }

  bus_ = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
  sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (!sink_) {
    NNDEPLOY_LOGE("Cannot find appsink\n");
    deinit();
    return base::kStatusCodeErrorInvalidParam;
  }

  // Set appsink properties for real-time streaming
  gst_app_sink_set_max_buffers(GST_APP_SINK(sink_), 1);
  gst_app_sink_set_drop(GST_APP_SINK(sink_), TRUE);

  GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    NNDEPLOY_LOGE("Failed to set stream pipeline to PLAYING\n");
    deinit();
    return base::kStatusCodeErrorInvalidParam;
  }

  // Probe first frame to get dimensions
  GstSample *first_sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
  if (first_sample) {
    GstCaps *caps = gst_sample_get_caps(first_sample);
    GstStructure *s = gst_caps_get_structure(caps, 0);
    gst_structure_get_int(s, "width", &width_gs_);
    gst_structure_get_int(s, "height", &height_gs_);
    // Re-push the sample by storing it for the next run()
    sample_ = first_sample;
  }

  width_ = width_gs_;
  height_ = height_gs_;
  size_ = INT_MAX;
  loop_count_ = size_;

  return base::kStatusCodeOk;
}

base::Status GStreamerStreamDecode::run() {
  if (!pipeline_ || !sink_) {
    return base::kStatusCodeErrorNullParam;
  }

  cv::Mat *mat = new cv::Mat();

  // Use pre-pulled sample if we saved one during setPath()
  if (sample_) {
    *mat = gstSampleToMat(sample_);
    gst_sample_unref(sample_);
    sample_ = nullptr;
  } else {
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
    if (!sample) {
      // Stream may have ended — attempt reconnection
      NNDEPLOY_LOGW("GStreamer stream sample is null, attempting reconnection...\n");
      delete mat;

      for (int attempt = 1; attempt <= reconnect_attempts_; ++attempt) {
        NNDEPLOY_LOGI("GStreamer reconnection attempt %d/%d\n", attempt, reconnect_attempts_);
        std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms_));

        deinit();
        if (setPath(path_) == base::kStatusCodeOk) {
          break;
        }
      }

      if (!pipeline_ || !sink_) {
        NNDEPLOY_LOGE("All GStreamer reconnection attempts failed for %s\n", path_.c_str());
        return base::kStatusCodeErrorInvalidParam;
      }

      cv::Mat *retry_mat = new cv::Mat();
      GstSample *retry_sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
      if (retry_sample) {
        *retry_mat = gstSampleToMat(retry_sample);
        gst_sample_unref(retry_sample);
      }
      outputs_[0]->set(retry_mat, false);
      return base::kStatusCodeOk;
    }

    *mat = gstSampleToMat(sample);
    gst_sample_unref(sample);
  }

  outputs_[0]->set(mat, false);
  index_++;
  return base::kStatusCodeOk;
}

// ==================== GStreamerStreamEncode ====================

base::Status GStreamerStreamEncode::init() {
  eos_ = false;
  pending_buffer_ = nullptr;
  return base::kStatusCodeOk;
}

base::Status GStreamerStreamEncode::deinit() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  if (source_) {
    gst_object_unref(source_);
    source_ = nullptr;
  }
  if (convert_) {
    gst_object_unref(convert_);
    convert_ = nullptr;
  }
  if (encode_) {
    gst_object_unref(encode_);
    encode_ = nullptr;
  }
  if (mux_) {
    gst_object_unref(mux_);
    mux_ = nullptr;
  }
  if (sink_) {
    gst_object_unref(sink_);
    sink_ = nullptr;
  }
  if (pending_buffer_) {
    gst_buffer_unref(pending_buffer_);
    pending_buffer_ = nullptr;
  }
  eos_ = false;
  return base::kStatusCodeOk;
}

base::Status GStreamerStreamEncode::setRefPath(const std::string &ref_path) {
  ref_path_ = ref_path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status GStreamerStreamEncode::setPath(const std::string &path) {
  path_ = path;
  path_changed_ = true;
  return base::kStatusCodeOk;
}

base::Status GStreamerStreamEncode::run() {
  cv::Mat *mat = inputs_[0]->getCvMat(this);
  if (!mat || mat->empty()) {
    NNDEPLOY_LOGE("Input mat is empty\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  return GStreamerVideoEncode::run();
}

// ==================== Factory Registration ====================

TypeCreatelDecodeRegister g_type_create_decode_node_register(
    base::kCodecTypeGStreamer, createGStreamerDecode);
TypeCreatelDecodeSharedPtrRegister
    g_type_create_decode_node_shared_ptr_register(base::kCodecTypeGStreamer,
                                                  createGStreamerDecodeSharedPtr);

Decode *createGStreamerDecode(base::CodecFlag flag, const std::string &name,
                              dag::Edge *output) {
  Decode *temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = new GStreamerImageDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagImages) {
    temp = new GStreamerImagesDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagVideo) {
    temp = new GStreamerVideoDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagCamera) {
    temp = new GStreamerCameraDecode(name, {}, {output}, flag);
  } else if (flag == base::kCodecFlagStreaming) {
    temp = new GStreamerStreamDecode(name, {}, {output}, flag);
  }
  return temp;
}

std::shared_ptr<Decode> createGStreamerDecodeSharedPtr(base::CodecFlag flag,
                                                       const std::string &name,
                                                       dag::Edge *output) {
  std::shared_ptr<Decode> temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = std::shared_ptr<GStreamerImageDecode>(
        new GStreamerImageDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagImages) {
    temp = std::shared_ptr<GStreamerImagesDecode>(
        new GStreamerImagesDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagVideo) {
    temp = std::shared_ptr<GStreamerVideoDecode>(
        new GStreamerVideoDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagCamera) {
    temp = std::shared_ptr<GStreamerCameraDecode>(
        new GStreamerCameraDecode(name, {}, {output}, flag));
  } else if (flag == base::kCodecFlagStreaming) {
    temp = std::shared_ptr<GStreamerStreamDecode>(
        new GStreamerStreamDecode(name, {}, {output}, flag));
  }
  return temp;
}

TypeCreatelEncodeRegister g_type_create_encode_node_register(
    base::kCodecTypeGStreamer, createGStreamerEncode);
TypeCreatelEncodeSharedPtrRegister
    g_type_create_encode_node_shared_ptr_register(base::kCodecTypeGStreamer,
                                                  createGStreamerEncodeSharedPtr);

Encode *createGStreamerEncode(base::CodecFlag flag, const std::string &name,
                               dag::Edge *input) {
  Encode *temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = new GStreamerImageEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagImages) {
    temp = new GStreamerImagesEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagVideo) {
    temp = new GStreamerVideoEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagCamera) {
    temp = new GStreamerCameraEncode(name, {input}, {}, flag);
  } else if (flag == base::kCodecFlagStreaming) {
    temp = new GStreamerStreamEncode(name, {input}, {}, flag);
  }
  return temp;
}

std::shared_ptr<Encode> createGStreamerEncodeSharedPtr(base::CodecFlag flag,
                                                       const std::string &name,
                                                       dag::Edge *input) {
  std::shared_ptr<Encode> temp = nullptr;
  if (flag == base::kCodecFlagImage) {
    temp = std::shared_ptr<GStreamerImageEncode>(
        new GStreamerImageEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagImages) {
    temp = std::shared_ptr<GStreamerImagesEncode>(
        new GStreamerImagesEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagVideo) {
    temp = std::shared_ptr<GStreamerVideoEncode>(
        new GStreamerVideoEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagCamera) {
    temp = std::shared_ptr<GStreamerCameraEncode>(
        new GStreamerCameraEncode(name, {input}, {}, flag));
  } else if (flag == base::kCodecFlagStreaming) {
    temp = std::shared_ptr<GStreamerStreamEncode>(
        new GStreamerStreamEncode(name, {input}, {}, flag));
  }
  return temp;
}

REGISTER_NODE("nndeploy::codec::GStreamerImageDecode", GStreamerImageDecode);
REGISTER_NODE("nndeploy::codec::GStreamerImagesDecode", GStreamerImagesDecode);
REGISTER_NODE("nndeploy::codec::GStreamerVideoDecode", GStreamerVideoDecode);
REGISTER_NODE("nndeploy::codec::GStreamerCameraDecode", GStreamerCameraDecode);
REGISTER_NODE("nndeploy::codec::GStreamerImageEncode", GStreamerImageEncode);
REGISTER_NODE("nndeploy::codec::GStreamerImagesEncode", GStreamerImagesEncode);
REGISTER_NODE("nndeploy::codec::GStreamerVideoEncode", GStreamerVideoEncode);
REGISTER_NODE("nndeploy::codec::GStreamerCameraEncode", GStreamerCameraEncode);
REGISTER_NODE("nndeploy::codec::GStreamerStreamDecode", GStreamerStreamDecode);
REGISTER_NODE("nndeploy::codec::GStreamerStreamEncode", GStreamerStreamEncode);

}  // namespace codec
}  // namespace nndeploy
