#ifndef _NNDEPLOY_DAG_DATA_PACKET_H_
#define _NNDEPLOY_DAG_DATA_PACKET_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API DataPacket {
 public:
  DataPacket() {}
  DataPacket(int &index, int &life) : index_(index), life_(life) {}

  void setIndex(int &index) { index_ = index; }
  int getIndex() { return index_; }

  void setLife(int &life) { life_ = life; }
  int getLife() { return life_; }

  // for single batch
  void setAnything(void *&anything) { anythings_.emplace_back(anything); }
  void *getAnything(int index) {
    if (index >= anythings_.size()) {
      return nullptr;
    }
    return anythings_[index];
  }

  // for multi batch
  void setAnything(std::vector<void *> &&anything) { anythings_ = anything; }
  std::vector<void *> &getAnything() { return anythings_; }

 private:
  int index_ = 0;
  int life_ = 1;
  std::vector<void *> anythings_;
#ifdef ENABLE_NNDEPLOY_OPENCV
  std::vector<cv::Mat *> cv_mats_;
#endif
};
}  // namespace dag

}  // namespace nndeploy

#endif