
VISION_CPU_INCLUDE(resize);
VISION_OPENCL_INCLUDE(resize);
VISION_CUDA_INCLUDE(resize);

namespace nndeploy {
namespace vision {

void resize(const device::Mat &src, device::Mat &dst, base::Size dsize,
            double inv_scale_x, double inv_scale_y, int interpolation) {
  VISION_CPU_RUN(resize);
  VISION_OPENCL_RUN(resize);
  VISION_CUDA_RUN(resize);           
}

}  // namespace vision
}  // namespace nndeploy