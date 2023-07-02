#ifndef B09E4B8F_561B_4518_BC33_E5C3D2B14BB1
#define B09E4B8F_561B_4518_BC33_E5C3D2B14BB1

using OpFunc = std::function<nndeploy::base::Status(
    std::vector<device::Tensor *> input, std::vector<device::Tensor *> output,
    std::shared_ptr<nndeploy::base::Param> param)>;

#endif /* B09E4B8F_561B_4518_BC33_E5C3D2B14BB1 */
