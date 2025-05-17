#include "nndeploy/device/opencl/opencl_device.h"
namespace nndeploy
{
  namespace device
  {
    uint8_t OpenCLDevice::getPlatformIdCounts()
    {
      clGetPlatformIDs(0, nullptr, &mPlatformIdCounts);
      if(mPlatformIdCounts <= 0)
      { return 0; }
      else{ return 1; }
    }
  } /* device */

} /* nndeploy */

int main() 
{
  auto d = nndeploy::device::OpenCLDevice();
  
  if(d.getPlatformIdCounts())
  {
    printf("OpenCL Platforms: %d", d.mPlatformIdCounts);
  }
  else
  {printf("failed to get OpenCL platforms");}
  return 0;
}