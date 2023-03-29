
#ifndef _NNDEPLOY_SOURCE_BASE_FILE_H_
#define _NNDEPLOY_SOURCE_BASE_FILE_H_

#include "nndeploy/source/base/include_c_cpp.h"
#include "nndeploy/source/base/macro.h"

namespace nndeploy {
namespace base {

extern NNDEPLOY_CC_API std::string openFile(const std::string &file_path);

extern NNDEPLOY_CC_API std::vector<std::string> getFileName(
    const std::string &dir_path, const std::string &suffix);

extern NNDEPLOY_CC_API std::string getExePath(
    const std::string &file_absolute_path);

extern NNDEPLOY_CC_API bool isDirectory(const std::string &dir_path);

extern NNDEPLOY_CC_API bool existDirectory(const std::string &dir_path);

extern NNDEPLOY_CC_API bool existFile(const std::string &file_path);

}  // namespace base
}  // namespace nndeploy

#endif