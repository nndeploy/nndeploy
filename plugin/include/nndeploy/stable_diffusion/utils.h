#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_UTILS_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_UTILS_H_

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace nndeploy {
namespace stable_diffusion {

// 美观的进度条插件封装成一个类
class ProgressBar {
 public:
  // 构造函数：total为总进度，barWidth为进度条宽度，prefix/suffix用于附加说明
  ProgressBar(int total, int barWidth = 80, const std::string &prefix = "",
              const std::string &suffix = "")
      : total(total),
        barWidth(barWidth),
        current(0),
        prefix(prefix),
        suffix(suffix),
        spinnerIndex(0) {
    spinnerChars = {"|", "/", "-", "\\"};
  }

  // 更新进度条显示，value为当前进度值
  void update(int value);

  // 结束显示，确保进度条显示满100%并换行
  void finish();

 private:
  int total;                              // 总进度值
  int barWidth;                           // 进度条宽度
  int current;                            // 当前进度值
  std::string prefix;                     // 前缀说明
  std::string suffix;                     // 后缀说明
  int spinnerIndex;                       // 旋转效果索引
  std::vector<std::string> spinnerChars;  // 旋转字符序列
};

}  // namespace stable_diffusion
}  // namespace nndeploy

#endif