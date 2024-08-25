
#ifndef _NNDEPLOY_NET_ASCEND_CL_ASCEND_CL_SESSION_H_
#define _NNDEPLOY_NET_ASCEND_CL_ASCEND_CL_SESSION_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/ascend_cl/ascend_cl_device.h"
#include "nndeploy/net/session.h"
#include "nndeploy/net/tensor_pool_1d.h"
#include "nndeploy/net/util.h"

namespace nndeploy {
namespace net {

class NNDEPLOY_CC_API AscendCLSession : public Session {
 public:
  AscendCLSession(const base::DeviceType &device_type);
  virtual ~AscendCLSession();

  virtual base::Status init(std::vector<TensorWrapper *> &tensor_repository,
                            std::vector<OpWrapper *> &op_repository,
                            bool is_dynamic_shape, base::ShapeMap max_shape);
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status preRun();
  virtual base::Status run();
  virtual base::Status postRun();

 protected:
  std::shared_ptr<TensorPool1DSharedObjectGreedyBySizeImprove> tensor_pool_;
  bool is_dynamic_shape_ = false;                // 是否是动态shape
  base::ShapeMap max_shape_ = base::ShapeMap();  // 当为动态输入时最大shape
  std::vector<TensorWrapper *> tensor_repository_;
  std::vector<OpWrapper *> op_repository_;
};

}  // namespace net
}  // namespace nndeploy

#endif
