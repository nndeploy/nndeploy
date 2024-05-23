# 新增一个设备

## 介绍

设备是nndeploy对硬件设备的抽象，通过对硬件设备的抽象，从而屏蔽不同硬件设备编程模型带来的差异性，nndeploy当前已经支持CPU、X86、ARM、CUDA、AscendCL等设备。主要功能如下

+ **统一的内存分配**：为不同设备提供统一的内存分配接口，从而可简化数据容器`Buffer`、`Mat`、`Tensor`的内存分配
  
+ **统一的内存拷贝**：为不同设备提供统一的内存拷贝接口（设备间拷贝、主从设备间上传/下载），从而可简化数据容器`Buffer`、`Mat`、`Tensor`的内存拷贝
  
+ **统一的同步操作**：为不同设备提供统一的同步操作接口，可简化设备端模型推理、算子等同步操作
  
+ **统一的硬件设备信息查询**：为不同设备提供统一的硬件设备信息查询接口，帮助用户更好的选择模型全流程部署的运行设备

## 步骤

新增一个设备主要分为以下三个步骤：

+ （1）新增设备类型枚举

+ （2）继承基类Architecture、继承基类Device

+ （3）修改cmake

### 步骤一：新增设备类型枚举

+ （1）修改文件 `<path>\include\nndeploy\base\common.h`，在`DeviceTypeCode`中添加新设备的枚举，格式为`kDeviceTypeCodeXxx` 

+ （2）修改文件 `<path>\source\nndeploy\base\common.cc`，在`DeviceTypeCode stringToDeviceTypeCode(const std::string &src)`函数中添加字符串转换为新设备的枚举实现

+ （3）修改文件 `<path>\include\nndeploy\base\status.h`，在`StatusCode`中添加新错误的枚举，格式为`kStatusCodeErrorDeviceXxx` 

### 步骤二： 继承基类Architecture、继承基类Device

#### 2.1 新增文件

+ （1）在`<path>\include\nndeploy\device`下新增`xxx\xxx_device.h`文件

+ （2）在`<path>\source\nndeploy\device`下新增`xxx\xxx_device.cc`文件

+ （3）[可选]在`<path>\source\nndeploy\device`下新增`xxx\xxx_include.h`文件，用于包含设备必要的头文件

#### 2.2 继承基类Architecture，实现XxxArchitecture

+ （1）在`<path>\include\nndeploy\device\xxx\xxx_device.h`下声明`XxxArchitecture`类，类似`<path>\include\nndeploy\device\cuda\cuda_device.h\CudaArchitecture`

  ```c++
  class CudaArchitecture : public Architecture {
   public:
    /**
     * @brief Construct a new Cuda Architecture object
     *
     * @param device_type_code
     */
    explicit CudaArchitecture(base::DeviceTypeCode device_type_code);

    /**
     * @brief Destroy the Cuda Architecture object
     *
     */
    virtual ~CudaArchitecture();

    /**
     * @brief Check whether the device corresponding to the current device id
     * exists, mainly serving GPU devices
     *
     * @param device_id - device id
     * @param command_queue - command_queue (corresponding to stream under CUDA,
     * corresponding to cl::command_queue under OpenCL)
     * @param library_path - Mainly serving OpenCL, using the OpenCL dynamic
     * library provided by the user
     * @return base::Status
     */
    virtual base::Status checkDevice(int device_id = 0,
                                     void *command_queue = nullptr,
                                     std::string library_path = "") override;

    /**
     * @brief Enable the device corresponding to the current device id，mainly
     * serving GPU devices
     *
     * @param device_id - device id
     * @param command_queue - command_queue (corresponding to stream under CUDA,
     * corresponding to cl::command_queue under OpenCL)
     * @param library_path - Mainly serving OpenCL, using the OpenCL dynamic
     * library provided by the user
     * @return base::Status
     */
    virtual base::Status enableDevice(int device_id = 0,
                                      void *command_queue = nullptr,
                                      std::string library_path = "") override;

    /**
     * @brief Get the Device object
     *
     * @param device_id
     * @return Device*
     */
    virtual Device *getDevice(int device_id) override;

    /**
     * @brief Get the Device Info object
     *
     * @param library_path
     * @return std::vector<DeviceInfo>
     */
    virtual std::vector<DeviceInfo> getDeviceInfo(
        std::string library_path = "") override;
  };  
  ```

+ （2）在`<path>\source\nndeploy\device\xxx\xxx_device.cc`中注册注新Architecture`TypeArchitectureRegister<XxxArchitecture> xxx_architecture_register(base::kDeviceTypeCodeXxx);`，类似`TypeArchitectureRegister<CudaArchitecture> cuda_architecture_register(base::kDeviceTypeCodeCuda);`中的实现

+ （3）在`<path>\source\nndeploy\device\xxx\xxx_device.cc`下实现`XxxArchitecture`，类似`<path>\source\nndeploy\device\cuda\cuda_device.cc\CudaArchitecture`中的实现

#### 2.3 继承基类Device，实现XxxDevice

+ （1）在`<path>\include\nndeploy\device\xxx\xxx_device.h`下声明`XxxDevice`类，类似`<path>\include\nndeploy\device\cuda\cuda_device.h\CudaDevice`
  ```c++
  class NNDEPLOY_CC_API CudaDevice : public Device {
    /**
     * @brief friend class
     *
     */
    friend class CudaArchitecture;  

   public:
    /**
     * @brief Convert MatDesc to BufferDesc.
     *
     * @param desc
     * @param config
     * @return BufferDesc
     */
    virtual BufferDesc toBufferDesc(const MatDesc &desc,
                                    const base::IntVector &config); 

    /**
     * @brief Convert TensorDesc to BufferDesc.
     *
     * @param desc
     * @param config
     * @return BufferDesc
     */
    virtual BufferDesc toBufferDesc(const TensorDesc &desc,
                                    const base::IntVector &config);
    /**
     * @brief Allocate Buffer
     *
     * @param size
     * @return Buffer*
     */
    virtual Buffer *allocate(size_t size);
    /**
     * @brief Allocate Buffer
     *
     * @param desc
     * @return Buffer*
     */
    virtual Buffer *allocate(const BufferDesc &desc);
    /**
     * @brief Deallocate buffer
     *
     * @param buffer
     */
    virtual void deallocate(Buffer *buffer);  

    /**
     * @brief Copy buffer
     *
     * @param src - Device's buffer.
     * @param dst - Device's buffer.
     * @return base::Status
     * @note Ensure that the memory space of dst is greater than or equal to src.
     */
    virtual base::Status copy(Buffer *src, Buffer *dst,
                            int index = 0);
    /**
     * @brief Download memory from the device to the host.
     *
     * @param src - Device's buffer.
     * @param dst - Host's buffer.
     * @return base::Status
     * @note Ensure that the memory space of dst is greater than or equal to src.
     */
    virtual base::Status download(Buffer *src, Buffer *dst,
                            int index = 0);
    /**
     * @brief Upload memory from the host to the device.
     *
     * @param src - Host's buffer.
     * @param dst - Device's buffer.
     * @return base::Status
     * @note Ensure that the memory space of dst is greater than or equal to src.
     */
    virtual base::Status upload(Buffer *src, Buffer *dst,
                            int index = 0);  

    /**
     * @brief synchronize
     *
     * @return base::Status
     */brancg
    virtual base::Status synchronize(); 

    /**
     * @brief Get the Command Queue object
     *
     * @return void*
     */
    virtual void *getCommandQueue();  

   protected:
    /**
     * @brief Construct a new Cuda Device object
     *
     * @param device_type
     * @param command_queue
     * @param library_path
     */
    CudaDevice(base::DeviceType device_type, void *command_queue = nullptr,
               std::string library_path = "")
        : Device(device_type), external_command_queue_(command_queue){};
    /**
     * @brief Destroy the Cuda Device object
     *
     */
    virtual ~CudaDevice(){};  

    /**
     * @brief init
     *
     * @return base::Status
     */
    virtual base::Status init();
    /**
     * @brief deinit
     *
     * @return base::Status
     */
    virtual base::Status deinit();  

   private:
    void *external_command_queue_ = nullptr;
    cudaStream_t stream_;
  };
  ``` 

  + （2） 在`<path>\source\nndeploy\device\xxx\xxx_device.cc`下实现`XxxDevice`，类似`<path>\source\nndeploy\device\cuda\cuda_device.cc\CudaDevice`中的实现


### 步骤三：修改cmake 

+ （1）修改主cmakelist `<path>\CMakeLists.txt`，
  + 新增设备编译选项`nndeploy_option(ENABLE_NNDEPLOY_DEVICE_XXX "ENABLE_NNDEPLOY_DEVICE_XXX" OFF)`
  + 由于新设备的增加，增加了源文件和头文件，需将源文件和头文件加入到编译文件中，需在`if(ENABLE_NNDEPLOY_DEVICE) endif()`的代码块中增加如下cmake源码
    ```shell
    if (ENABLE_NNDEPLOY_DEVICE_XXX)
      file(GLOB_RECURSE DEVICE_XXX_SOURCE
        "${ROOT_PATH}/include/nndeploy/device/xxx/*.h"
        "${ROOT_PATH}/source/nndeploy/device/xxx/*.cc"
      )
      set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_XXX_SOURCE})
    endif()
    ```

+ （2）[可选]如果需要链接设备相关的三方库
  + 需要在`<path>\cmake`目录下新增`xxx.cmake`，类似`<path>\cmake\ascend_cl.cmake`或`<path>\cmake\cuda.cmake`
  + 修改`<path>\cmake\nndeploy.cmake`，新增`include("${ROOT_PATH}/cmake/xxx.cmake")`

+ （3）修改`<path>\build\config.cmake`,新增设备编译选项`set(ENABLE_NNDEPLOY_DEVICE_XXX ON)`


