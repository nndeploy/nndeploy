
# AI硬件信息获取 - 用于推理优化和部署决策
##--精度类型支持 (Precision Types)--
# fp32, fp16, bf16, tf32, fp8(E4M3/E5M2), int8, int4, int2, int1等
##--计算性能 (Compute Performance)--
# GFLOPS/TFLOPS
##--内存系统 (Memory System)--
# 内存带宽(GB/s)
# 内存容量(GB)
# 缓存层级(L1/L2/L3)
# 内存类型(GDDR6/6X, HBM2/3)
##--硬件架构 (Hardware Architecture)--
# 核心数量
# 计算单元数
# 张量核心数
##--驱动和软件栈 (Software Stack)--
# 驱动版本, CUDA版本, OpenCL版本, ROCm版本
# 编译器版本(NVCC, HIP, SYCL)
# 运行时版本(CUDA Runtime, ROCm Runtime, SYCL Runtime)
# 深度学习框架兼容性, 算子库版本(cuDNN, MIOpen)
##--互连和通信 (Interconnect)--
# PCIe版本和带宽(3.0/4.0/5.0)
# NVLink速度和拓扑
# InfiniBand, Ethernet, 片间互连带宽
# P2P内存访问, RDMA支持

# get cpu info

# get cuda info

# get opencl info

# get syscl info

# get metal info

# get vulkan info

# get opengl info

# get rocm info