
#ifndef _NNDEPLOY_INCLUDE_BASE_FILE_H_
#define _NNDEPLOY_INCLUDE_BASE_FILE_H_

#include "nndeploy/include/base/include_c_cpp.h"

namespace nndeploy {
namespace base {

std::string openFile(const std::string &file_path);

std::vector<std::string> getFileName(const std::string &dir_path,
                                     const std::string &suffix);

std::string getExePath(const std::string &file_absolute_path);

bool isDirectory(const std::string &dir_path);

bool existDirectory(const std::string &dir_path);

bool existFile(const std::string &file_path);

}  // namespace base
}  // namespace nndeploy

#endif