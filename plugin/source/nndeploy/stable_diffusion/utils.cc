#include "nndeploy/stable_diffusion/utils.h"

namespace nndeploy {
namespace stable_diffusion {

void ProgressBar::update(int value) {
  current = value;
  double fraction = static_cast<double>(current) / total;
  int pos = static_cast<int>(barWidth * fraction);

  // 获取当前旋转符号，并更新下一个的索引
  std::string spinner = spinnerChars[spinnerIndex];
  spinnerIndex = (spinnerIndex + 1) % spinnerChars.size();

  std::cout << "\r" << prefix << " [";
  // 输出进度条内容：用“█”表示已完成部分，用“░”表示剩余部分
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "█";
    else
      std::cout << "░";
  }
  std::cout << "] " << int(fraction * 100) << "% " << spinner << " " << suffix;
  std::cout.flush();
}

void ProgressBar::finish() {
  update(total);
  std::cout << std::endl;
}

}  // namespace stable_diffusion
}  // namespace nndeploy