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
#ifndef _NNCORE_INCLUDE_GRAPH_CONFIG_H_
#define _NNCORE_INCLUDE_GRAPH_CONFIG_H_

#include "nncore/include/base/log.h"
#include "nncore/include/base/macro.h"
#include "nncore/include/base/object.h"
#include "nncore/include/base/status.h"
#include "nncore/include/base/type.h"

namespace nncore {
namespace graph {

class Config {
 public:
 private:
  std::string name_;
};

}  // namespace graph
}  // namespace nncore

#endif