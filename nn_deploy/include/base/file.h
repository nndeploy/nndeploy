/**
 * @file file.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-20
 *
 * @copyright Copyright (c) 2022
 * @todo
 * # 像python一样操作
 *
 */
#ifndef _NN_DEPLOY_BASE_FILE_
#define _NN_DEPLOY_BASE_FILE_

#include "nn_deploy/base/include_c_cpp.h"

namespace nn_deploy {
namespace base {

std::string OpenFile(const std::string &file_path);

std::vector<std::string> GetFileName(const std::string &dir_path,
                                     const std::string &suffix);

std::string GetExePath();

bool IsDirectory(const std::string &dir_path);

bool ExistDirectory(const std::string &dir_path);

bool ExistFile(const std::string &file_path);

// filename_functionname_line
std::string GenDefaultStrTag();

}  // namespace base
}  // namespace nn_deploy

#endif