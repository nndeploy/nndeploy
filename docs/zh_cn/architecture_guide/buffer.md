# Buffer

我们现在来看一下 Buffer 的设计实现。Buffer 所在的目录在 framework/include/nndeploy/device/buffer.h 里面，源文件在对应 source 目录下。

还是延续我们的原则，先看成员变量，再看成员函数。

首先看最重要的成员变量 data_，这是具体持有的数据，比如说是 malloc 出来的 data，new 出来的 data，这个具体的数据。如果是 CPU 设备，那就是 CPU data；如果是 GPU 设备，那就是 GPU data。

那怎么来标明这个 data 是什么设备的呢？这里有 device_ 成员变量来指明。比如说通过华为昇腾 NPU 分配出来的 data 是什么样子的，这个就由 device_ 来标识。

假如配置了 CPU 内存池、NPU 内存池，那我们从这个内存池分配出来的数据，就由 memory_pool_ 这个成员变量来标识。

然后是描述这段内存的信息，比如内存空间有多大？ buffer_desc_ 这个成员变量来描述的。

这个 Buffer 的数据指针也可以是外部传入的，比如说我现在有一个 Buffer，我可以把外部的指针传给我这个 Buffer 来使用，就像 OpenCV 的 Mat 可以传入一个外部数据指针一样。

那这个内存就有类型，比如是外部传入的，是内部分配的，对于 OpenCL 这种还有内存映射、共享内存等类型，这是由 memory_type_ 这个成员变量来标识的。

Buffer 也支持拷贝构造函数和赋值构造函数。同样的，它也需要引用计数 ref_count_，比如说两个 Buffer 共享同一块 data，当它引用计数等于 0 的时候，这个 data 才可以被释放。

然后我们跳进去看一下 BufferDesc 是如何描述一个 Buffer 的。

uffer 会有大小，size_ 表示它有多大，是一个 SizeVector 类型。为什么是一个 Vector 呢？比如说我们通过 malloc 分配出来的内存是一个一维的大小，
但是对于一些异构设备，比如说 OpenCL，它有一种内存对象叫 CL_image_2D，这种对象它是有高和宽的，还有通道数，所以它需要多维来描述它的大小，size表示他的逻辑大小

还会有一个 real_size_，就是具体分配的真实大小，这个real_size始终都不会发生变化。

BUFFER需要支持 Tensor 的 reshape 操作，Tensor 真正持有数据的是靠 Buffer。假如说一开始这个 Tensor 是 1×1024×1024，然后 reshape 为了 1×512×512，那它的逻辑大小肯定是发生变化的，但真实大小是不会发生变化的。

然后是 config_，就是有些内存它有特殊的要求，比如说通过 CUDA 的 cudaMalloc 分配内存时有一些特殊参数。例如，在CUDA中我们可能需要分配固定内存（PINNED MEMORY），这种内存不会被操作系统分页，可以实现更高效的主机和设备之间的数据传输。我们可以通过在config_中设置相应的标志来指定这种内存类型, 这样内存对象就有了具体的属性和要求。

好了，我们看完 BufferDesc 之后，再来看一下buffer。

<!-- 首先 Buffer 可以传入一个大小来构造，因为 Tensor 会经常用到 Buffer，所以 BufferDesc 是一个常用的数据结构。Buffer 类有一大堆方便的构造函数，包括拷贝构造函数。

我们还提供了一些帮助函数，比如返回大小（getSize），返回整个 size vector（getSizeVector），返回真实的大小（getRealSize），然后判断内存属性是否相同（isSameMemoryType），判断维度是否相同（isSameDims），判断是否为空（empty）等等。

有时候我们需要比较两个 Buffer 的内存大小，所以有一些比较函数。在开发过程中，我们可能需要调试 Buffer，因为它存放着实际数据，在编写算子或前后处理过程中需要查看 Buffer 内容，所以我们提供了打印函数（print）。

当 Tensor reshape 之后，对应的 Buffer size 也要发生变化，所以我们有 resize 函数，可以仅仅修改逻辑大小而不改变真实大小。 -->

我们来看Tensor的成员函数。

Buffer类提供了丰富的成员函数，以支持各种内存操作和管理需求。

### 构造函数

Buffer类提供了多种构造方式，以适应不同的内存分配和管理场景：有基于基于设备的构造、使用外部指针的构造、指定内存类型的构造、基于内存池的构造、拷贝和移动构造

<!-- 1. **基于设备的构造**：
   - `Buffer(Device *device, size_t size)`：从指定设备分配指定大小的内存
   - `Buffer(Device *device, const BufferDesc &desc)`：根据BufferDesc描述从设备分配内存

2. **使用外部指针的构造**：
   - `Buffer(Device *device, size_t size, void *ptr)`：使用外部提供的指针，关联到指定设备
   - `Buffer(Device *device, const BufferDesc &desc, void *ptr)`：使用外部指针，并按BufferDesc描述关联到设备

3. **指定内存类型的构造**：
   - `Buffer(Device *device, size_t size, void *ptr, base::MemoryType memory_type)`：使用外部指针，并指定内存类型
   - `Buffer(Device *device, const BufferDesc &desc, void *ptr, base::MemoryType memory_type)`：使用外部指针，按BufferDesc描述并指定内存类型

4. **基于内存池的构造**：
   - `Buffer(MemoryPool *memory_pool, size_t size)`：从内存池分配指定大小的内存
   - `Buffer(MemoryPool *memory_pool, const BufferDesc &desc)`：根据BufferDesc从内存池分配内存

5. **拷贝和移动构造**：
   - 拷贝构造函数`Buffer(const Buffer &buffer)`：创建一个Buffer的浅拷贝，共享底层数据
   - 赋值操作符`Buffer &operator=(const Buffer &buffer)`：赋值操作，实现浅拷贝
   - 移动构造函数`Buffer(Buffer &&buffer) noexcept`：移动构造，转移资源所有权
   - 移动赋值操作符`Buffer &operator=(Buffer &&buffer) noexcept`：移动赋值，转移资源所有权 -->

这些构造函数使得Buffer能够灵活地适应各种内存分配和管理场景，无论是直接从设备分配内存，还是使用预分配的内存，或者从内存池获取内存。

### set函数

Buffer类提供了模板函数`set<T>(T value)`，用于将Buffer中的所有元素设置为指定值。

<!-- 这个函数的实现考虑了不同设备类型的特性： -->

<!-- 1. 首先检查Buffer是否为空，如果为空则返回错误
2. 对于主机设备（如CPU），直接操作内存
3. 对于非主机设备（如GPU），先在主机上分配临时内存，设置值后再上传到设备内存
4. 根据元素类型T和Buffer大小计算元素数量，然后设置每个元素的值
5. 对于非主机设备，完成操作后释放临时内存

这个函数在初始化Buffer内容时非常有用，例如将权重初始化为零或特定值。 -->

### 克隆和拷贝函数

Buffer提供了两个重要的数据复制函数：

1. **clone()**：创建一个完全独立的Buffer副本，包括复制底层数据。这是深拷贝操作，新Buffer与原Buffer完全独立。

2. **copyTo(Buffer *dst)**：将当前Buffer的数据复制到目标Buffer。这个函数会处理不同设备间的数据传输，例如从CPU复制到GPU或从一个GPU复制到另一个GPU。

这些函数在需要数据备份或在不同设备间传输数据时非常有用。

### 序列化函数

Buffer支持序列化和反序列化操作，从而来支持模型权重的序列化和反序列化：

<!-- 1. **serialize(std::string &bin_str)**：将Buffer的内容序列化到输出流。这个函数会保存Buffer的描述信息和实际数据。

2. **serialize(const std::string &bin_str)**：从输入流反序列化数据到Buffer。这个函数会根据流中的信息重建Buffer。

3. **serializeToSafetensors**：当启用SAFETENSORS_CPP支持时，Buffer可以序列化为safetensors格式，这是一种更安全、更高效的张量存储格式。

这些序列化功能使得Buffer能够方便地保存到文件或从文件加载，支持模型的持久化存储和分发。 -->

### 打印函数

Buffer提供了`print(std::ostream &stream)`函数，用于将Buffer的信息输出到指定流（默认为标准输出）。这个函数会输出：

1. Buffer的基本信息，如大小、设备类型、内存类型等
2. 对于较小的Buffer，可能还会输出实际数据内容

这个函数在调试过程中非常有用，可以帮助开发者检查Buffer的状态和内容。

### 帮助函数

Buffer类提供了一系列帮助函数，例如描述信息获取、数据方法、打印等等函数， 使得操作和查询Buffer更加方便：

<!-- 1. **内存管理函数**：
   - `justModify(const size_t &size)`：仅修改逻辑大小，不改变实际分配的内存
   - `justModify(const base::SizeVector &size)`：使用向量指定新的逻辑大小
   - `justModify(const BufferDesc &desc)`：使用新的BufferDesc更新Buffer描述

2. **状态查询函数**：
   - `empty()`：检查Buffer是否为空
   - `getDeviceType()`：获取设备类型
   - `getDevice()`：获取关联的设备
   - `getMemoryPool()`：获取关联的内存池
   - `isMemoryPool()`：检查是否使用内存池

3. **描述信息获取**：
   - `getDesc()`：获取BufferDesc
   - `getSize()`：获取逻辑大小
   - `getSizeVector()`：获取逻辑大小向量
   - `getRealSize()`：获取实际分配的大小
   - `getRealSizeVector()`：获取实际大小向量
   - `getConfig()`：获取配置信息

4. **数据访问**：
   - `getData()`：获取数据指针
   - `getMemoryType()`：获取内存类型

5. **引用计数管理**：
   - `addRef()`：增加引用计数
   - `subRef()`：减少引用计数 -->

这些帮助函数使得Buffer的使用更加灵活和方便，能够满足各种内存管理和数据操作的需求。

好了，有关我们 buffer 的实现，我们就讲到这里了
