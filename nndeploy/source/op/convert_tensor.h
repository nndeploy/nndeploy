#ifndef B09E4B8F_561B_4518_BC33_E5C3D2B14BB1
#define B09E4B8F_561B_4518_BC33_E5C3D2B14BB1

base::Status convertTensor(device::Tensor *src, device::Tensor *dst,
                           std::vector<float> scale, std::vectot<float> bias);

base::Status convertTensor(device::Mat *src, device::Tensor *dst,
                           std::vector<float> scale, std::vectot<float> bias);

#endif /* B09E4B8F_561B_4518_BC33_E5C3D2B14BB1 */
