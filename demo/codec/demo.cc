
#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"

using namespace nndeploy;

int main(int argc, char const *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, const_cast<char***>(&argv), true);
  base::TimeProfiler profiler;
  if (demo::timeProfile()) {
    profiler.start("nndeploy::demo::codec");
  }

  base::CodecType codec_type = demo::getCodecType();
  base::CodecFlag codec_flag = demo::getCodecFlag();
  std::string input_path = demo::getInputPath();
  std::string output_path = demo::getOutputPath();

  if (codec_type == base::kCodecTypeNone) {
    NNDEPLOY_LOGE("Invalid --codec_type. Use: kCodecTypeOpenCV, kCodecTypeFFmpeg, kCodecTypeGStreamer\n");
    return -1;
  }

  dag::Edge *output = new dag::Edge("output");
  dag::Edge *input = new dag::Edge("input");

  if (codec_flag == base::kCodecFlagImage) {
    // Decode single image
    codec::Decode *decode = codec::createDecode(codec_type, codec_flag, "decode", output);
    if (!decode) {
      NNDEPLOY_LOGE("Failed to create decode node\n");
      return -1;
    }
    decode->setPath(input_path);
    decode->init();
    base::Status status = decode->run();
    NNDEPLOY_LOGI("Image decode: width=%d, height=%d, status=%d\n",
                  decode->getWidth(), decode->getHeight(), status);

    // Encode single image
    codec::Encode *encode = codec::createEncode(codec_type, codec_flag, "encode", input);
    if (!encode) {
      NNDEPLOY_LOGE("Failed to create encode node\n");
      return -1;
    }
    encode->setPath(output_path);
    encode->init();
    // Set the decoded frame as input manually
    cv::Mat *frame = output->getCvMat(decode);
    if (frame) {
      input->set(frame, false);
      status = encode->run();
      NNDEPLOY_LOGI("Image encode status=%d\n", status);
    }
    delete decode;
    delete encode;

  } else if (codec_flag == base::kCodecFlagVideo) {
    // Video decode (limited frames for demo)
    codec::Decode *decode = codec::createDecode(codec_type, codec_flag, "decode", output);
    if (!decode) {
      NNDEPLOY_LOGE("Failed to create video decode node\n");
      return -1;
    }
    decode->setPath(input_path);
    decode->init();
    NNDEPLOY_LOGI("Video: width=%d, height=%d, fps=%.2f, total_frames=%d\n",
                  decode->getWidth(), decode->getHeight(), decode->getFps(), decode->getSize());

    int frame_count = std::min(decode->getSize(), 30);
    for (int i = 0; i < frame_count; i++) {
      decode->run();
      cv::Mat *frame = output->getCvMat(decode);
      if (frame && !frame->empty()) {
        NNDEPLOY_LOGI("  decoded frame %d/%d: %dx%d\n", i + 1, frame_count, frame->cols, frame->rows);
      }
    }
    delete decode;

  } else if (codec_flag == base::kCodecFlagStreaming) {
    // Streaming decode (reads a few frames, then stops)
    codec::Decode *decode = codec::createDecode(codec_type, codec_flag, "decode", output);
    if (!decode) {
      NNDEPLOY_LOGE("Failed to create streaming decode node\n");
      return -1;
    }
    decode->setPath(input_path);
    decode->init();
    NNDEPLOY_LOGI("Streaming: width=%d, height=%d, infinite frames\n",
                  decode->getWidth(), decode->getHeight());

    int frame_count = 0;
    int max_frames = 30;
    while (frame_count < max_frames) {
      base::Status status = decode->run();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Stream read error at frame %d\n", frame_count);
        break;
      }
      cv::Mat *frame = output->getCvMat(decode);
      if (frame && !frame->empty()) {
        NNDEPLOY_LOGI("  stream frame %d: %dx%d\n", frame_count + 1, frame->cols, frame->rows);
      }
      frame_count++;
    }
    NNDEPLOY_LOGI("Read %d streaming frames from %s\n", frame_count, input_path.c_str());
    delete decode;

  } else if (codec_flag == base::kCodecFlagCamera) {
    // Camera decode (reads a few frames, then stops)
    codec::Decode *decode = codec::createDecode(codec_type, codec_flag, "decode", output);
    if (!decode) {
      NNDEPLOY_LOGE("Failed to create camera decode node\n");
      return -1;
    }
    decode->setPath(input_path.empty() ? "0" : input_path);
    decode->init();
    NNDEPLOY_LOGI("Camera: width=%d, height=%d\n", decode->getWidth(), decode->getHeight());

    int frame_count = 0;
    int max_frames = 15;
    while (frame_count < max_frames) {
      base::Status status = decode->run();
      if (status != base::kStatusCodeOk) break;
      cv::Mat *frame = output->getCvMat(decode);
      if (frame && !frame->empty()) {
        NNDEPLOY_LOGI("  camera frame %d: %dx%d\n", frame_count + 1, frame->cols, frame->rows);
      }
      frame_count++;
    }
    NNDEPLOY_LOGI("Captured %d camera frames\n", frame_count);
    delete decode;

  } else {
    NNDEPLOY_LOGE("Unsupported --codec_flag for this demo. "
                  "Use: kCodecFlagImage, kCodecFlagVideo, kCodecFlagCamera, kCodecFlagStreaming\n");
    return -1;
  }

  if (demo::timeProfile()) {
    profiler.end("nndeploy::demo::codec");
    profiler.print();
  }

  NNDEPLOY_LOGI("Codec demo completed successfully\n");
  return 0;
}
