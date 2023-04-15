
/**
 * @TODO:
 * 1. <sys/stat.h>
 * # 这个头文件是否是跨平台的？
 * # 不是跨平台的，需要改成跨平台的
 * 2. 检查函数并修复
 *
 */
#include "nndeploy/source/base/file.h"

#include <sys/stat.h>

namespace nndeploy {
namespace base {

std::string openFile(const std::string &file_path) {
  std::string result = "";
  std::ifstream fin(file_path, std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    return "";
  }
  fin.seekg(0, std::ios::end);
  result.resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(result.at(0)), result.size());
  fin.close();
  return result;
}

// TODO: fix this function
std::vector<std::string> getFileName(const std::string &dir_path,
                                     const std::string &suffix) {
  std::vector<std::string> fileNames;
  // _finddata_t data;
  // auto handle = _findfirst((dir_path + "/*." + suffix).c_str(), &data);
  // if (handle == -1) {
  //   return fileNames;
  // }
  // do {
  //   std::string s1 = data.name;
  //   if (data.attrib & _A_SUBDIR) {
  //     continue;
  //   } else {
  //     std::string s2 = "." + suffix;
  //     if (s1.rfind(s1) == s1.size() - s1.size()) {
  //       fileNames.push_back(s1);
  //     }
  //   }
  // } while (_findnext(handle, &data) == 0);
  // _findclose(handle);
  return fileNames;
}

std::string getExePath(const std::string &file_absolute_path) {
  std::string::size_type pos = file_absolute_path.rfind("/");
  if (pos == std::string::npos) {
    return "/";
  } else {
    return file_absolute_path.substr(0, pos + 1);
  }
}

bool isDirectory(const std::string &dir_path) {
  struct stat s;
  if (stat(dir_path.c_str(), &s) == 0 && s.st_mode & S_IFDIR) {
    return true;
  } else {
    return false;
  }
}

bool existDirectory(const std::string &dir_path) {
  return isDirectory(dir_path);
}

bool existFile(const std::string &file_path) {
  struct stat s;
  if (stat(file_path.c_str(), &s) == 0 && s.st_mode & S_IFREG) {
    return true;
  } else {
    return false;
  }
}

std::string pathJoin(const std::vector<std::string> &paths,
                     const std::string &sep) {
  if (paths.size() == 1) {
    return paths[0];
  }
  std::string filepath = "";
  for (const auto &path : paths) {
    if (filepath == "") {
      filepath += path;
      continue;
    }
    if (path[0] == sep[0] || filepath.back() == sep[0]) {
      filepath += path;
    } else {
      filepath += sep + path;
    }
  }
  return filepath;
}

std::string pathJoin(const std::string &folder, const std::string &filename,
                     const std::string &sep) {
  return pathJoin(std::vector<std::string>{folder, filename}, sep);
}

}  // namespace base
}  // namespace nndeploy