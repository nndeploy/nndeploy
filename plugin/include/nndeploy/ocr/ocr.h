
#ifndef _NNDEPLOY_OCR_CLASSIFICATION_H_
#define _NNDEPLOY_OCR_CLASSIFICATION_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvtcolor_bn.h"
#include "nndeploy/preprocess/params.h"
#include "nndeploy/ocr/result.h"

namespace nndeploy {
namespace orc {


// class NNDEPLOY_CC_API OCRClassificationPostProcess : public dag::Node {
//  public:
//   OCRClassificationPostProcess(const std::string &name) : dag::Node(name) {
//     key_ = "nndeploy::ocr::OCRClassificationPostProcess";
//     this->setInputTypeInfo<device::Tensor>();
//     this->setOutputTypeInfo<OCRResult>();
//   }
//   OCRClassificationPostProcess(const std::string &name,
//                             std::vector<dag::Edge *> inputs,
//                             std::vector<dag::Edge *> outputs)
//       : dag::Node(name, inputs, outputs) {
//     key_ = "nndeploy::orc::OCRClassificationPostProcess";
//     this->setInputTypeInfo<device::Tensor>();
//     this->setOutputTypeInfo<OCRResult>();
//   }
//   virtual ~OCRClassificationPostProcess() {}

//   virtual base::Status run();
// };

}  // namespace OCRClassification
}  // namespace nndeploy

#endif /* _NNDEPLOY_OCR_CLASSIFICATION_H_ */