
#include "nndeploy/base/file.h"

#include "nndeploy/base/log.h"

#if NNDEPLOY_OS_WINDOWS
#undef NOMINMAX
#define NOMINMAX
#include <direct.h>
#include <errno.h>
#include <io.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <windows.h>
#elif NNDEPLOY_OS_UNIX
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#if NNDEPLOY_OS_WINDOWS
static const char k_g_native_separator = '\\';
const char k_g_dir_separators[] = "/\\";
namespace {
struct dirent {
  const char* d_name;
};

struct DIR {
#if defined(WINRT) || defined(_WIN32_WCE)
  WIN32_FIND_DATAW data;
#else
  WIN32_FIND_DATAA data;
#endif
  HANDLE handle;
  dirent ent;
#ifdef WINRT
  DIR() {}
  ~DIR() {
    if (ent.d_name) delete[] ent.d_name;
  }
#endif
};

DIR* opendir(const char* path) {
  DIR* dir = new DIR;
  dir->ent.d_name = 0;
#if defined(WINRT) || defined(_WIN32_WCE)
  std::string full_path = std::string(path) + "\\*";
  wchar_t wfull_path[MAX_PATH];
  size_t copied = mbstowcs(wfull_path, full_path.c_str(), MAX_PATH);
  assert((copied != MAX_PATH) && (copied != (size_t)-1));
  dir->handle = ::FindFirstFileExW(wfull_path, FindExInfoStandard, &dir->data,
                                   FindExSearchNameMatch, nullptr, 0);
#else
  dir->handle = ::FindFirstFileExA((std::string(path) + "\\*").c_str(),
                                   FindExInfoStandard, &dir->data,
                                   FindExSearchNameMatch, nullptr, 0);
#endif
  if (dir->handle == INVALID_HANDLE_VALUE) {
    /*closedir will do all cleanup*/
    delete dir;
    return 0;
  }
  return dir;
}

dirent* readdir(DIR* dir) {
#if defined(WINRT) || defined(_WIN32_WCE)
  if (dir->ent.d_name != 0) {
    if (::FindNextFileW(dir->handle, &dir->data) != TRUE) return 0;
  }
  size_t asize = wcstombs(nullptr, dir->data.cFileName, 0);
  assert((asize != 0) && (asize != (size_t)-1));
  char* aname = new char[asize + 1];
  aname[asize] = 0;
  wcstombs(aname, dir->data.cFileName, asize);
  dir->ent.d_name = aname;
#else
  if (dir->ent.d_name != 0) {
    if (::FindNextFileA(dir->handle, &dir->data) != TRUE) return 0;
  }
  dir->ent.d_name = dir->data.cFileName;
#endif
  return &dir->ent;
}

void closedir(DIR* dir) {
  ::FindClose(dir->handle);
  delete dir;
}

}  // namespace
#elif NNDEPLOY_OS_UNIX
static const char k_g_native_separator = '/';
const char k_g_dir_separators[] = "/";
#endif

namespace nndeploy {
namespace base {

static bool isDir(const std::string& path, DIR* dir) {
#if defined _WIN32 || defined _WIN32_WCE
  DWORD attributes;
  BOOL status = TRUE;
  if (dir)
    attributes = dir->data.dwFileAttributes;
  else {
    WIN32_FILE_ATTRIBUTE_DATA all_attrs;
#if defined WINRT || defined _WIN32_WCE
    wchar_t wpath[MAX_PATH];
    size_t copied = mbstowcs(wpath, path.c_str(), MAX_PATH);
    assert((copied != MAX_PATH) && (copied != (size_t)-1));
    status = ::GetFileAttributesExW(wpath, GetFileExInfoStandard, &all_attrs);
#else
    status =
        ::GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &all_attrs);
#endif
    attributes = all_attrs.dwFileAttributes;
  }

  return status && ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0);
#else
  struct stat stat_buf;
  if (0 != stat(path.c_str(), &stat_buf)) return false;
  int is_dir = S_ISDIR(stat_buf.st_mode);
  return is_dir != 0;
#endif
}

static bool wildcmp(const char* string, const char* wild) {
  // Based on wildcmp written by Jack Handy - <A
  // href="mailto:jakkhandy@hotmail.com">jakkhandy@hotmail.com</A>
  const char *cp = 0, *mp = 0;

  while ((*string) && (*wild != '*')) {
    if ((*wild != *string) && (*wild != '?')) {
      return false;
    }

    wild++;
    string++;
  }

  while (*string) {
    if (*wild == '*') {
      if (!*++wild) {
        return true;
      }

      mp = wild;
      cp = string + 1;
    } else if ((*wild == *string) || (*wild == '?')) {
      wild++;
      string++;
    } else {
      wild = mp;
      string = cp++;
    }
  }

  while (*wild == '*') {
    wild++;
  }

  return *wild == 0;
}

static void globRec(const std::string& directory, const std::string& wildchart,
                    std::vector<std::string>& result, bool recursive,
                    bool includeDirectories, const std::string& pathPrefix) {
  DIR* dir;

  if ((dir = opendir(directory.c_str())) != 0) {
    /* find all the files and directories within directory */
    try {
      struct dirent* ent;
      while ((ent = readdir(dir)) != 0) {
        const char* name = ent->d_name;
        if ((name[0] == 0) || (name[0] == '.' && name[1] == 0) ||
            (name[0] == '.' && name[1] == '.' && name[2] == 0))
          continue;

        std::string path = joinPath(directory, name);
        std::string entry = joinPath(pathPrefix, name);

        if (isDir(path, dir)) {
          if (recursive)
            globRec(path, wildchart, result, recursive, includeDirectories,
                    entry);
          if (!includeDirectories) continue;
        }

        if (wildchart.empty() || wildcmp(name, wildchart.c_str()))
          result.push_back(entry);
      }
    } catch (...) {
      closedir(dir);
      throw;
    }
    closedir(dir);
  } else {
    NNDEPLOY_LOGE("could not open directory: %s.\n", directory.c_str());
  }
}

std::string openFile(const std::string& file_path) {
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

bool isPathSeparator(char c) { return c == '/' || c == '\\'; }

bool exists(const std::string& path) {
#if defined _WIN32 || defined WINCE
  BOOL status = TRUE;
  {
    WIN32_FILE_ATTRIBUTE_DATA all_attrs;
#ifdef WINRT
    wchar_t wpath[MAX_PATH];
    size_t copied = mbstowcs(wpath, path.c_str(), MAX_PATH);
    assert((copied != MAX_PATH) && (copied != (size_t)-1));
    status = ::GetFileAttributesExW(wpath, GetFileExInfoStandard, &all_attrs);
#else
    status =
        ::GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &all_attrs);
#endif
  }

  return !!status;
#else
  struct stat stat_buf;
  return (0 == stat(path.c_str(), &stat_buf));
#endif
}

bool isDirectory(const std::string& path) { return isDir(path, nullptr); }

void removeAllFile(const std::string& path) {
  if (!exists(path)) return;
  if (isDirectory(path)) {
    std::vector<std::string> entries;
    glob(path, std::string(), entries, false, true);
    for (size_t i = 0; i < entries.size(); i++) {
      const std::string& e = entries[i];
      removeAllFile(e);
    }
#ifdef _MSC_VER
    bool result = _rmdir(path.c_str()) == 0;
#else
    bool result = rmdir(path.c_str()) == 0;
#endif
    if (!result) {
      NNDEPLOY_LOGE("Can't remove directory: %s.\n", path.c_str());
    }
  } else {
#ifdef _MSC_VER
    bool result = _unlink(path.c_str()) == 0;
#else
    bool result = unlink(path.c_str()) == 0;
#endif
    if (!result) {
      NNDEPLOY_LOGE("Can't remove file: %s.\n", path.c_str());
    }
  }
}

std::string getcwd() {
  std::array<char, 4096> buf;
#if defined WIN32 || defined _WIN32 || defined WINCE
#ifdef WINRT
  return std::string();
#else
  DWORD sz = GetCurrentDirectoryA(0, nullptr);
  sz = GetCurrentDirectoryA((DWORD)buf.size(), buf.data());
  return std::string(buf.data(), static_cast<size_t>(sz));
#endif
#elif NNDEPLOY_OS_UNIX
  for (;;) {
    char* p = ::getcwd(buf.data(), buf.size());
    if (p == nullptr) {
      if (errno == ERANGE) {
        continue;
      }
      return std::string();
    }
    break;
  }
  return std::string(buf.data(), static_cast<size_t>(strlen(buf.data())));
#else
  return std::string();
#endif
}

std::string canonicalPath(const std::string& path) {
  std::string result;
#ifdef _WIN32
  const char* result_str = _fullpath(nullptr, path.c_str(), 0);
#else
  const char* result_str = realpath(path.c_str(), nullptr);
#endif
  if (result_str) {
    result = std::string(result_str);
    free((void*)result_str);
  }
  return result.empty() ? path : result;
}

std::string joinPath(const std::string& base, const std::string& path) {
  if (base.empty()) return path;
  if (path.empty()) return base;

  bool base_sep_flag = isPathSeparator(base[base.size() - 1]);
  bool path_sep_flag = isPathSeparator(path[0]);
  std::string result;
  if (base_sep_flag && path_sep_flag) {
    result = base + path.substr(1);
  } else if (!base_sep_flag && !path_sep_flag) {
    result = base + NNDEPLOY_PATH_SEP + path;
  } else {
    result = base + path;
  }
  return result;
}

std::string getParentPath(const std::string& path) {
  std::string::size_type loc = path.find_last_of("/\\");
  if (loc == std::string::npos) return std::string();
  return std::string(path, 0, loc);
}

std::wstring getParentPath(const std::wstring& path) {
  std::wstring::size_type loc = path.find_last_of(L"/\\");
  if (loc == std::wstring::npos) return std::wstring();
  return std::wstring(path, 0, loc);
}

void glob(std::string pattern, std::vector<std::string>& result,
          bool recursive) {
  result.clear();
  std::string path, wildchart;

  if (isDir(pattern, 0)) {
    if (strchr(k_g_dir_separators, pattern[pattern.size() - 1]) != 0) {
      path = pattern.substr(0, pattern.size() - 1);
    } else {
      path = pattern;
    }
  } else {
    size_t pos = pattern.find_last_of(k_g_dir_separators);
    if (pos == std::string::npos) {
      wildchart = pattern;
      path = ".";
    } else {
      path = pattern.substr(0, pos);
      wildchart = pattern.substr(pos + 1);
    }
  }

  globRec(path, wildchart, result, recursive, false, path);
  std::sort(result.begin(), result.end());
}

void glob(const std::string& directory, const std::string& pattern,
          std::vector<std::string>& result, bool recursive,
          bool includeDirectories) {
  globRec(directory, pattern, result, recursive, includeDirectories, directory);
  std::sort(result.begin(), result.end());
}

void globRelative(const std::string& directory, const std::string& pattern,
                  std::vector<std::string>& result, bool recursive,
                  bool includeDirectories) {
  globRec(directory, pattern, result, recursive, includeDirectories,
          std::string());
  std::sort(result.begin(), result.end());
}

bool createDirectory(const std::string& path) {
#if defined WIN32 || defined _WIN32 || defined WINCE
#ifdef WINRT
  wchar_t wpath[MAX_PATH];
  size_t copied = mbstowcs(wpath, path.c_str(), MAX_PATH);
  assert((copied != MAX_PATH) && (copied != (size_t)-1));
  int result = CreateDirectoryA(wpath, nullptr) ? 0 : -1;
#else
  int result = _mkdir(path.c_str());
#endif
#elif NNDEPLOY_OS_UNIX
  int result = mkdir(path.c_str(), 0777);
#else
  int result = -1;
#endif

  if (result == -1) {
    return isDirectory(path);
  }
  return true;
}

bool createDirectories(const std::string& path_) {
  std::string path = path_;
  for (;;) {
    char last_char = path.empty() ? 0 : path[path.length() - 1];
    if (isPathSeparator(last_char)) {
      path = path.substr(0, path.length() - 1);
      continue;
    }
    break;
  }

  if (path.empty() || path == "./" || path == ".\\" || path == ".") return true;
  if (isDirectory(path)) return true;

  size_t pos = path.rfind('/');
  if (pos == std::string::npos) pos = path.rfind('\\');
  if (pos != std::string::npos) {
    std::string parent_directory = path.substr(0, pos);
    if (!parent_directory.empty()) {
      if (!createDirectories(parent_directory)) return false;
    }
  }

  return createDirectory(path);
}

}  // namespace base
}  // namespace nndeploy