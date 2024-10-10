
#ifndef _NNDEPLOY_BASE_FILE_H_
#define _NNDEPLOY_BASE_FILE_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"

namespace nndeploy {
namespace base {

#ifdef _MSC_VER
#define NNDEPLOY_PATH_SEP "\\"
#else
#define NNDEPLOY_PATH_SEP "/"
#endif

extern NNDEPLOY_CC_API std::string openFile(const std::string &file_path);

extern NNDEPLOY_CC_API bool isPathSeparator(char c);

extern NNDEPLOY_CC_API bool exists(const std::string &path);
extern NNDEPLOY_CC_API bool isDirectory(const std::string &path);

extern NNDEPLOY_CC_API void removeAllFile(const std::string &path);

extern NNDEPLOY_CC_API std::string getcwd();

/**
 * @brief Converts path p to a canonical absolute path
 * Symlinks are processed if there is support for them on running platform.
 *
 * @param path input path. Target file/directory should exist.
 */
extern NNDEPLOY_CC_API std::string canonicalPath(const std::string &path);

/** Join path components */
extern NNDEPLOY_CC_API std::string joinPath(const std::string &base,
                                            const std::string &path);

/** Get parent directory */
extern NNDEPLOY_CC_API std::string getParentPath(const std::string &path);
extern NNDEPLOY_CC_API std::wstring getParentPath(const std::wstring &path);

/**
 * Generate a list of all files that match the globbing pattern.
 *
 * Result entries are prefixed by base directory path.
 *
 * @param directory base directory
 * @param pattern filter pattern (based on '*'/'?' symbols). Use empty string to
 * disable filtering and return all results
 * @param[out] result result of globing.
 * @param recursive scan nested directories too
 * @param include_directories include directories into results list
 */
extern NNDEPLOY_CC_API void glob(const std::string &directory,
                                 const std::string &pattern,
                                 std::vector<std::string> &result,
                                 bool recursive = false,
                                 bool include_directories = false);

/**
 * Generate a list of all files that match the globbing pattern.
 *
 * @param directory base directory
 * @param pattern filter pattern (based on '*'/'?' symbols). Use empty string to
 * disable filtering and return all results
 * @param[out] result globbing result with relative paths from base directory
 * @param recursive scan nested directories too
 * @param include_directories include directories into results list
 */
extern NNDEPLOY_CC_API void globRelative(const std::string &directory,
                                         const std::string &pattern,
                                         std::vector<std::string> &result,
                                         bool recursive = false,
                                         bool include_directories = false);

extern NNDEPLOY_CC_API bool createDirectory(const std::string &path);
extern NNDEPLOY_CC_API bool createDirectories(const std::string &path_param);

}  // namespace base
}  // namespace nndeploy

#endif