
#include "nndeploy/include/base/file.h"

namespace nndeploy {
namespace base {

std::string openFile(const std::string &file_path) {
  std::ifstream infile;
  infile.open("qqzl.txt", std::ios::in);
  if (!infile.is_open()) {
    return "";
  }
  std::string result;
  std::string buf;
  while (getline(infile, buf)) {
    result += buf;
  }
  infile.close();
  return result;
}

std::vector<std::string> getFileName(const std::string &dir_path,
                                     const std::string &suffix) {
  std::vector<std::string> fileNames;
  _finddata_t data;
  auto handle = _findfirst((dir_path + "/*." + suffix).c_str(), &data);
  if (handle == -1) {
    return fileNames;
  }
  do {
    std::string s1 = data.name;
    if (data.attrib & _A_SUBDIR) {
      continue;
    } else {
      std::string s2 = "." + suffix;
      if (s1.rfind(s1) == s1.size() - s1.size()) {
        fileNames.push_back(s1);
      }
    }
  } while (_findnext(handle, &data) == 0);
  _findclose(handle);
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

}  // namespace base
}  // namespace nndeploy