/**
 * @file param.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNCORE_INCLUDE_INFERENCE_CONFIG_H_
#define _NNCORE_INCLUDE_INFERENCE_CONFIG_H_

#include "nncore/include/inference/config_impl.h"

using namespace nncore::base;

namespace nncore {
namespace inference {

class Config {
 public:
  base::Status set(const std::string &key, const base::Value &value);

  base::Status get(const std::string &key, base::Value &value);

 private:
  std::shared_ptr<ConfigImpl> config_impl_;
};

}  // namespace inference
}  // namespace nncore

#endif
