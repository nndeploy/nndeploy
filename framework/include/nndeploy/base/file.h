
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

// /**
//  * 将字符串写入输出流
//  *
//  * @param output 输出流
//  * @param str 要写入的字符串
//  * @return base::Status 写入操作的状态
//  */
// extern NNDEPLOY_CC_API base::Status writeString(std::ostream &output,
//                                                 const std::string &str) {
//   // 写入字符串长度
//   uint32_t length = str.length();
//   output.write(reinterpret_cast<const char *>(&length), sizeof(uint32_t));

//   // 写入字符串内容
//   output.write(str.c_str(), length);

//   // 检查写入是否成功
//   if (output.good()) {
//     return base::kStatusCodeOk;
//   } else {
//     return base::kStatusCodeErrorIO;
//   }
// }

// /**
//  * 从输入流读取字符串
//  *
//  * @param input 输入流
//  * @param str 用于存储读取的字符串
//  * @return base::Status 读取操作的状态
//  */
// extern NNDEPLOY_CC_API base::Status readString(std::istream &input,
//                                                std::string &str) {
//   // 读取字符串长度
//   uint32_t length;
//   input.read(reinterpret_cast<char *>(&length), sizeof(uint32_t));

//   // 检查是否成功读取长度
//   if (!input.good()) {
//     return base::kStatusCodeErrorIO;
//   }

//   // 读取字符串内容
//   str.resize(length);
//   input.read(&str[0], length);

//   // 检查读取是否成功
//   if (input.good()) {
//     return base::kStatusCodeOk;
//   } else {
//     return base::kStatusCodeErrorIO;
//   }
// }

// /**
//  * 将向量写入输出流
//  *
//  * @param output 输出流
//  * @param vec 要写入的向量
//  * @param write_func 用于写入单个元素的函数
//  * @return base::Status 写入操作的状态
//  */
// template <typename T>
// extern NNDEPLOY_CC_API base::Status writeVector(
//     std::ostream &output, const std::vector<T> &vec,
//     std::function<base::Status(std::ostream &, const T &)> write_func) {
//   // 写入向量大小
//   uint32_t size = static_cast<uint32_t>(vec.size());
//   output.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));

//   // 检查写入大小是否成功
//   if (!output.good()) {
//     return base::kStatusCodeErrorIO;
//   }

//   // 写入每个元素
//   for (const auto& item : vec) {
//     base::Status status = write_func(output, item);
//     if (status != base::kStatusCodeOk) {
//       return status;
//     }
//   }

//   // 检查整体写入是否成功
//   if (output.good()) {
//     return base::kStatusCodeOk;
//   } else {
//     return base::kStatusCodeErrorIO;
//   }

// }

// /**
//  * 将映射写入输出流
//  *
//  * @param output 输出流
//  * @param map 要写入的映射
//  * @param write_func 用于写入键值对的函数
//  * @return base::Status 写入操作的状态
//  */
// template <typename K, typename V>
// extern NNDEPLOY_CC_API base::Status writeMap(
//     std::ostream &output, const std::map<K, V> &map,
//     std::function<base::Status(std::ostream &, const std::pair<K, V> &)>
//         write_func);

}  // namespace base
}  // namespace nndeploy

#endif