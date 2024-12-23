#ifndef _NNDEPLOY_CODEC_CODEC_H_
#define _NNDEPLOY_CODEC_CODEC_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API DecodeNode : public dag::Node {
 public:
  DecodeNode(base::CodecFlag flag, const std::string &name, dag::Edge *output)
      : dag::Node(name, nullptr, output), flag_(flag) {}
  virtual ~DecodeNode() {}

  base::CodecFlag getCodecFlag() { return flag_; }
  void setPath(const std::string &path) { path_ = path; }

  void setSize(int size) {
    if (size > 0) {
      size_ = size;
    }
  }
  int getSize() { return size_; }

  double getFps() { return fps_; };
  int getWidth() { return width_; };
  int getHeight() { return height_; };

  virtual base::EdgeUpdateFlag updataInput() {
    if (index_ < size_) {
      return base::kEdgeUpdateFlagComplete;
    } else {
      return base::kEdgeUpdateFlagTerminate;
    }
  }

  virtual base::Status run() = 0;

 protected:
  base::CodecFlag flag_ = base::kCodecFlagImage;
  std::string path_ = "";
  int size_ = 0;
  double fps_ = 0.0;
  int width_ = 0;
  int height_ = 0;
  int index_ = 0;
};

class NNDEPLOY_CC_API EncodeNode : public dag::Node {
 public:
  EncodeNode(base::CodecFlag flag, const std::string &name, dag::Edge *input)
      : dag::Node(name, input, nullptr), flag_(flag){};
  virtual ~EncodeNode(){};

  base::CodecFlag getCodecFlag() { return flag_; }
  void setPath(const std::string &path) { path_ = path; }
  void setRefPath(const std::string &ref_path) { ref_path_ = ref_path; }
  void setFourcc(const std::string &fourcc) { fourcc_ = fourcc; }
  void setFps(double fps) { fps_ = fps; };
  void setWidth(int width) { width_ = width; };
  void setHeight(int height) { height_ = height; };

  int getIndex() { return index_; };

  virtual base::Status run() = 0;

 protected:
  base::CodecFlag flag_ = base::kCodecFlagImage;
  std::string path_ = "";
  std::string ref_path_ = "";
  std::string fourcc_ = "MJPG";
  double fps_ = 0.0;
  int width_ = 0;
  int height_ = 0;
  int index_ = 0;
};

using createDecodeNodeFunc = std::function<DecodeNode *(
    base::CodecFlag flag, const std::string &name, dag::Edge *output)>;

std::map<base::CodecType, createDecodeNodeFunc>
    &getGlobaCreatelDecodeNodeFuncMap();

class TypeCreatelDecodeNodeRegister {
 public:
  explicit TypeCreatelDecodeNodeRegister(base::CodecType type,
                                         createDecodeNodeFunc func) {
    getGlobaCreatelDecodeNodeFuncMap()[type] = func;
  }
};

extern NNDEPLOY_CC_API DecodeNode *createDecodeNode(base::CodecType type,
                                                    base::CodecFlag flag,
                                                    const std::string &name,
                                                    dag::Edge *output);

using createEncodeNodeFunc = std::function<EncodeNode *(
    base::CodecFlag flag, const std::string &name, dag::Edge *input)>;

std::map<base::CodecType, createEncodeNodeFunc>
    &getGlobaCreatelEncodeNodeFuncMap();

class TypeCreatelEncodeNodeRegister {
 public:
  explicit TypeCreatelEncodeNodeRegister(base::CodecType type,
                                         createEncodeNodeFunc func) {
    getGlobaCreatelEncodeNodeFuncMap()[type] = func;
  }
};

extern NNDEPLOY_CC_API EncodeNode *createEncodeNode(base::CodecType type,
                                                    base::CodecFlag flag,
                                                    const std::string &name,
                                                    dag::Edge *input);

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_CODEC_H_ */
