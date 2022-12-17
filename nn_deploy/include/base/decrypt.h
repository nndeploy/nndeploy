/**
 * @file decrypt.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NN_DEPLOY_BASE_DECRYPT_
#define _NN_DEPLOY_BASE_DECRYPT_

#include "nn_deploy/base/include_c_cpp.h"

namespace nn_deploy {
namespace base {

std::string DecryptFromPath(const std::string &src, const std::string *license);

std::string DecryptFromBuffer(const std::string &src, const std::string *license);

}  // namespace base
}  // namespace nn_deploy

#endif