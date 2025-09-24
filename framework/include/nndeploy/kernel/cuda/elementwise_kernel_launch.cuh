#ifndef _NNDEPLOY_KERNEL_CUDA_ELEMENTWISE_KERNEL_LAUNCH_H_
#define _NNDEPLOY_KERNEL_CUDA_ELEMENTWISE_KERNEL_LAUNCH_H_

namespace nndeploy {
namespace kernel {

constexpr int kBlockSize = 256;

constexpr int kNumWaves = 32;

// 计算执行核函数所需的线程块数量
inline cudaError_t GetNumBlocks(int64_t n, int* num_blocks) {
  int dev;  // 当前GPU设备编号
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int sm_count;  // GPU的流式多处理器(SM)数量
  {
    cudaError_t err =
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int tpm;  //每个SM支持的最大线程数
  {
    cudaError_t err = cudaDeviceGetAttribute(
        &tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  // 计算线程块数量：取两个值中的较小者
  // 1. 至少需要的线程块数：(n + 255) / 256（向上取整）
  // 2. 最大合理线程块数：根据SM数量和每个SM的线程数计算
  *num_blocks = std::max<int>(
      1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                           sm_count * tpm / kBlockSize * kNumWaves));
  return cudaSuccess;
}

//获取打包数据的类型
template <typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T),
                                             pack_size * sizeof(T)>::type;
};

template <typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

// 用于打包多个元素
template <typename T, int pack_size>
union Pack {
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
  __device__ Pack() {}
  PackType<T, pack_size> storage;
  T elem[pack_size];
};

template <typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed {
  __device__ Packed() {}
  union {
    T elem[pack_size];
  };
};

// 最大打包字节数：128位（16字节），GPU一次可高效访问的内存大小
constexpr int kMaxPackBytes = 128 / 8;
// 最大打包元素数量：最多8个元素
constexpr int kMaxPackSize = 8;

constexpr int Min(int a, int b) { return a < b ? a : b; }

// 计算单个类型的最佳打包大小
template <typename T>
constexpr int PackSize() {
  // 取最大字节限制和最大元素数量中的较小值
  return Min(kMaxPackBytes / sizeof(T), kMaxPackSize);
}

// 计算多个类型的最佳打包大小（取所有类型的最小值）
template <typename T, typename U, typename... Args>
constexpr int PackSize() {
  return Min(PackSize<T>(), PackSize<U, Args...>());
}

// 检查FunctorT是否有Apply2方法
template <typename T>
class HasApply2 {
  typedef char one;  // 辅助类型，大小为1字节
  struct two {       // 辅助结构体，大小为2字节
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::Apply2));

  template <typename C>
  static two test(...);

 public:
  // 根据测试结果判断是否有Apply2方法（1字节则有，2字节则无）
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

// 对打包数据应用操作（当有Apply2方法且pack_size为偶数时）
template <int pack_size, typename FunctorT, typename Dst, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == true &&
                                       pack_size % 2 == 0,
                                   Packed<Dst, pack_size>>::type
ApplyPack(const FunctorT& functor, const Packed<IN, pack_size>... in) {
  Packed<Dst, pack_size> ret;
#pragma unroll
  // 每次处理2个元素（利用Apply2优化）
  for (int j = 0; j < pack_size; j += 2) {
    functor.Apply2(ret.elem + j, (in.elem + j)...);
  }
  return ret;
}

template <int pack_size, typename FunctorT, typename Dst, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == false ||
                                       pack_size % 2 != 0,
                                   Packed<Dst, pack_size>>::type
ApplyPack(const FunctorT& functor, const Packed<IN, pack_size>... in) {
  Packed<Dst, pack_size> ret;
#pragma unroll
  // 逐个处理每个元素
  for (int j = 0; j < pack_size; ++j) {
    ret.elem[j] = functor((in.elem[j])...);
  }
  return ret;
}

template <int pack_size, typename FactoryT, typename Dst, typename... IN>
__global__ void __launch_bounds__(kBlockSize)
    ApplyGeneric(FactoryT factory, int64_t n_pack,
                 Packed<Dst, pack_size>* pack_r,
                 const Packed<IN, pack_size>*... pack_in, int64_t n_tail,
                 Dst* tail_r, const IN*... tail_in) {
  auto functor = factory();

  const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;

  // 每个线程处理一个数据包
  for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x) {
    pack_r[i] = ApplyPack<pack_size, decltype(functor), Dst, IN...>(
        functor, (pack_in[i])...);
  }

  // 处理尾部未打包的元素（当总数不是pack_size的倍数时）
  if (global_tid < n_tail) {
    tail_r[global_tid] = functor((tail_in[global_tid])...);
  }
}

template <typename FunctorT>
struct SimpleFactory {
  explicit SimpleFactory(FunctorT functor) : functor_(functor) {}
  __device__ FunctorT operator()() const { return functor_; }

 private:
  FunctorT functor_;
};

template <size_t pack_size>
bool IsAlignedForPack() {
  return true;
}

//检查所有指针是否满足打包对齐要求
template <size_t pack_size, typename T, typename... Args>
bool IsAlignedForPack(const T* ptr, const Args*... others) {
  return reinterpret_cast<uintptr_t>(ptr) % sizeof(Pack<T, pack_size>) == 0 &&
         IsAlignedForPack<pack_size, Args...>(others...);
}

template <size_t pack_size, typename FactoryT, typename Dst, typename... IN>
cudaError_t LaunchKernel(FactoryT factory, int64_t n, Dst* r, const IN*... in,
                         cudaStream_t stream) {
  const int64_t n_pack = n / pack_size;
  const int64_t tail_offset = n_pack * pack_size;  // 尾部数据的起始位置
  const int64_t n_tail = n - tail_offset;  // 尾部未打包的数据数量
  int num_blocks;

  // 获取线程块数量
  {
    cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
    if (err != cudaSuccess) {
      return err;
    }
  }

  ApplyGeneric<pack_size, FactoryT, Dst, IN...>
      <<<num_blocks, kBlockSize, 0, stream>>>(
          factory, n_pack, reinterpret_cast<Packed<Dst, pack_size>*>(r),
          (reinterpret_cast<const Packed<IN, pack_size>*>(in))..., n_tail,
          r + tail_offset, (in + tail_offset)...);

  return cudaPeekAtLastError();
}

// 通用启动器，根据对齐情况选择最佳打包策略
template <typename FactoryT, typename Dst, typename... IN>
struct GenericLauncher {
  static cudaError_t Launch(FactoryT factory, int64_t n, Dst* r,
                            const IN*... in, cudaStream_t stream) {
    constexpr int max_pack_size = PackSize<Dst, IN...>();

    // 如果内存对齐，使用最大打包大小；否则使用1（不打包）
    if (IsAlignedForPack<max_pack_size, Dst, IN...>(r, in...)) {
      return LaunchKernel<max_pack_size, FactoryT, Dst, IN...>(factory, n, r,
                                                               in..., stream);
    } else {
      return LaunchKernel<1, FactoryT, Dst, IN...>(factory, n, r, in...,
                                                   stream);
    }
  }
};

template <typename FactoryT, typename Dst, typename Src>
inline cudaError_t UnaryFactory(FactoryT factory, int64_t n, Dst* dst,
                                const Src* src, cudaStream_t stream) {
  return GenericLauncher<FactoryT, Dst, Src>::Launch(factory, n, dst, src,
                                                     stream);
}

template <typename FunctorT, typename Dst, typename Src>
inline cudaError_t UnaryLaunch(FunctorT functor, int64_t n, Dst* dst,
                               const Src* src, cudaStream_t stream) {
  return UnaryFactory(SimpleFactory<FunctorT>(functor), n, dst, src, stream);
}
}  // namespace kernel
}  // namespace nndeploy

#endif /*  */
