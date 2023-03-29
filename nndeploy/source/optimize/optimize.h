

// #ifndef _NNDEPLOY_SOURCE_OPTIMIZE_OPTIMIZE_H_
// #define _NNDEPLOY_SOURCE_OPTIMIZE_OPTIMIZE_H_

// #include "nndeploy/source/base/basic.h"
// #include "nndeploy/source/base/log.h"
// #include "nndeploy/source/base/macro.h"
// #include "nndeploy/source/base/object.h"
// #include "nndeploy/source/base/status.h"
// #include "nndeploy/source/ir/model.h"
// #include "nndeploy/source/interpret/interpret.h"

// namespace nndeploy {
// namespace optimize {

// /**
//  * @brief
//  * # convert edit quantize forward 都要依赖这个模块
//  */
// class Optimize {
//  public:
//   Optimize();
//   ~Optimize();

//   virtual base::Status init(std::vector<std::string> config, base::ShapeMap
//   static_shape = base::ShapeMap()) = 0; virtual base::Status deinit() = 0;

//   std::vector<std::string> getOptimizePass();

//   base::Status getStaticShape(base::ShapeMap shape_map);

//   ir::Model *run(ir::Model *model);

//  private:
//   // 委托给ir::Model
//   base::Status inferShape(base::ShapeMap static_shape);

//  protected:
//   std::vector<std::string> config_;
//   base::ShapeMap static_shape_ = base::ShapeMap();
// };

// std::vector<std::string> getAllConvertOptPass();

// std::vector<std::string> getAllInterpretOptPass();

// std::vector<std::string> getAllOptPass();

// }  // namespace optimize
// }  // namespace nndeploy

// #endif