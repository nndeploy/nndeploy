/**
 * @file file.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-21
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNCORE_INCLUDE_BASE_FILE_H_
#define _NNCORE_INCLUDE_BASE_FILE_H_

#include "nncore/include/base/include_c_cpp.h"

namespace nncore {
namespace base {

std::string openFile(const std::string &file_path);

std::vector<std::string> getFileName(const std::string &dir_path,
                                     const std::string &suffix);

std::string getExePath();

bool isDirectory(const std::string &dir_path);

bool existDirectory(const std::string &dir_path);

bool existFile(const std::string &file_path);

/**
 * @brief
 *
 * @return std::string -> _file_function_line_
 */
std::string genDefaultStrTag();

}  // namespace base
}  // namespace nncore

#endif