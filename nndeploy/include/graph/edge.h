/**
 * @file device.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-24
 *
 * @copyright Copyright (c) 2022
 * @note ref opencv
 */
#ifndef _NNDEPLOY_INCLUDE_GRAPH_EDGE_H_
#define _NNDEPLOY_INCLUDE_GRAPH_EDGE_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/device/mat.h"
#include "nndeploy/include/inference/tensor.h"


namespace nndeploy {
namespace graph {

class Edge {
 public:
 private:
  std::string name_;
};

}  // namespace graph
}  // namespace nndeploy

#endif