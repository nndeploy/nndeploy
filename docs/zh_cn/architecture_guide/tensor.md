

我们先来看一下 Tensor 的具体实现。Tensor 代码在 framework/include/device/tensor.h，它的源文件在对应的 source 目录下。

我们还是延续我们讲代码的老思路，先看成员变量。再看成员函数。

首先是 Tensor 需要名字，在上一节 IR 中知道到，ir中算子的输入输出由唯一标识符name标识，到了模型推理阶段，会编程Tenosr，因此在一个模型内部，不会出现同名的tensor。然后是描述 Tensor 的 TensorDesc，这里会描述 Tensor 具体有多大，是什么类型，是什么格式。然后是这个 Tensor 的数据是外部的还是内部的，当拷贝构造函数的时候，它需要有引用计数，当引用计数为修改为 0 的时候，这个 buffer 就可以销毁了。buffer 是真正持有数据的成员变量。

好的，我们现在跳转到 TensorDesc 再去看一下，具体该如何描述一个 Tensor 还需要哪些信息？我们跳到 TensorDesc 再去看一下这个具体的数据结构。它代码在 framework/include/type 目录下，打开type.h，一样的，先看一下它的成员变量。

首先是数据类型（data_type_），它定义了Tensor中存储的数据是什么类型。比如在代码中我们可以看到默认是float类型：`base::DataType data_type_ = base::dataTypeOf<float>()`。当然，Tensor也可以存储其他类型的数据，如int8、uint8、int32、float16等。

然后是数据格式（data_format_），它描述了Tensor中数据的排列方式。默认值是`base::kDataFormatNotSupport`，表示初始状态下没有指定格式。常见的格式有NCHW和NHWC，其中N表示batch size（批次大小），C表示channel（通道数），H和W分别表示height（高度）和width（宽度）。例如，对于一张RGB图像，如果使用NCHW格式，那么数据排列顺序是先按照通道（R、G、B）排列，再按照高度和宽度排列；而如果使用NHWC格式，则是先按照高度和宽度排列，再按照通道排列。不同的深度学习框架可能默认使用不同的格式，如PyTorch常用NCHW，而TensorFlow常用NHWC。

第三个是形状（shape_），它是一个IntVector类型的成员变量，用于描述Tensor的维度大小。例如，对于一个批次大小为1、通道数为3、高度为224、宽度为224的图像Tensor，其shape可以表示为[1, 3, 224, 224]（NCHW格式）。形状决定了Tensor中包含多少个元素，以及这些元素如何在各个维度上分布。

最后是步长（stride_）。
它是一个SizeVector类型的成员变量。
当我们需要访问Tensor中的特定元素时，stride告诉我们如何计算内存偏移量。
这对于高效地遍历和操作Tensor数据非常重要。
例如，对于一个形状为[2, 3, 4]的Tensor，如果它在内存中是连续存储的，那么默认的步长可能是[12, 4, 1]。
这表示在第一个维度上移动一步需要跳过12个元素。
在第二个维度上移动一步需要跳过4个元素。
在第三个维度上移动一步需要跳过1个元素。
步长的设计使得我们可以高效地访问Tensor中的元素。

好了，我们 TensorDesc 的成员变量讲完了，TensorDesc 是服务于 Tensor 的，我们希望 Tensor 可以构造起来很简单，因此为 TensorDesc 提供了 多 个构造函数，比如说什么都没有的构造函数，传参传 data type、传 format、传 shape 的构造函数。复制构造函数和析构函数。判断两个 TensorDesc 是否相等的帮助函数，Tensor 需要支持权重的序列化和反序列化，那TensorDesc 也需要有序列化和反序列化的函数。

Tensor作为非常作用的数据结构，经常需要调试它，TensorDesc 提供了一个打印函数，可以把这些信息通过字符串的形式打印出来。

我们 TensorDesc 就讲完了，我们再回到 Tensor。

再来看下is_external_这个成员变量，它主要是说明buffer是Tensor内部构造的，还是外部传入的，内部构造的很好理解，我们来看看外部构造的场景，在推理框架中，Tensor存在基于图的内存共享机制，多个Tensor就会共享一个buffer，通常做法是，外部统一管理这些buffer，然后把buffer传入给tensor，这个是，成员变量is_external_就是ture.


引用计数（ref_count_）也是与buffer相关的重要机制。当调用拷贝构造函数时，由于Tensor通常持有大量数据，我们希望通过浅拷贝实现以提高效率，这样buffer就会被多个tensor共享。在这种情况下，我们需要引用计数机制来正确管理Tensor的内存释放。每当创建一个新的Tensor副本时，引用计数加一；当Tensor被销毁时，引用计数减一。只有当引用计数降为零时，才真正释放buffer占用的内存资源。这种机制确保了内存管理的安全性和效率。

关于is_external_和ref_count_这两个成员变量，虽然它们都与Tensor的内存管理有关，但它们的作用是不同的哈：

is_external_主要标识buffer的所有权问题。当is_external_为true时，表示buffer是从外部传入的，Tensor不负责创建和销毁这个buffer，只是使用它。这种情况通常出现在以下场景：
1. 用户直接提供了预分配的内存空间
2. 多个Tensor共享同一块内存（如图推理中的内存优化）
3. 与其他系统集成时，需要使用外部系统提供的内存

而ref_count_则是解决多个Tensor实例共享同一个buffer时的内存管理问题。它通过计数机制确保只有当最后一个引用该buffer的Tensor被销毁时，才真正释放buffer的内存。这主要用于以下场景：
1. Tensor的拷贝构造和赋值操作（浅拷贝）
2. 函数返回Tensor对象时
3. 将Tensor存储在容器中时

两者的主要区别在于：
- is_external_决定了buffer的生命周期是否由Tensor管理
- ref_count_决定了何时释放由Tensor管理的buffer

在实际使用中，当is_external_为true时，无论ref_count_如何变化，Tensor都不会尝试释放buffer；而当is_external_为false时，只有当ref_count_降为0时，Tensor才会释放buffer。这种设计使得Tensor能够灵活地适应各种内存管理场景，既可以自主管理内存，也可以使用外部内存。

最后是buffer_成员变量，它是真正存储Tensor数据的地方。Buffer类封装了底层内存分配和管理的细节，使Tensor能够在不同设备（如CPU、GPU）上高效地存储和访问数据。

我们来看Tensor的成员函数。

首先看看构造函数，Tensor提供了多种构造方式以适应不同场景。最基本的构造函数包括默认构造函数`Tensor()`、只指定名称的构造函数`Tensor(const std::string &name)`以及指定TensorDesc的构造函数`Tensor(const TensorDesc &desc, const std::string &name)`。这些构造函数不会分配实际的内存空间，只是设置Tensor的基本属性。

对于不分配内存的成员函数，除了上述构造函数外，还有`Tensor(const TensorDesc &desc, Buffer *buffer, const std::string &name)`，它允许用户直接传入一个已有的Buffer，这在需要共享内存或使用外部预分配内存的场景中非常有用。

分配内存的构造函数主要有两类：基于Device和基于MemoryPool的。Device方式直接从设备申请内存，而MemoryPool方式则从内存池中获取内存，后者通常能提供更高效的内存管理。


再来看一下析构函数`~Tensor()`。当Tensor对象被销毁时，析构函数会被调用。在析构函数中，首先会检查引用计数`ref_count_`。如果引用计数不为空，则将其减一；当引用计数减为零时，表示没有其他Tensor对象引用该buffer，此时会根据`is_external_`的值决定是否释放buffer。如果`is_external_`为false，表示buffer是由Tensor内部创建的，此时会释放buffer占用的内存；如果为true，则不会释放buffer，因为buffer的所有权不属于Tensor。

接着是对动态形状推理的支持，tensor的`reshape`函数允许在不改变底层数据的情况下修改Tensor的形状。
关于`reshape`函数的实现，它有三种情况需要考虑：
1. 如果buffer为空，直接修改TensorDesc中的shape即可，不需要其他操作。
2. 如果buffer不为空，且reshape后的buffer空间小于或等于当前buffer的实际空间空间，则可以直接修改shape并更新buffer的描述信息。
3. 如果buffer不为空，且reshape后需要的空间大于当前buffer的空间，这种情况下就直接报错，为什么选择直接报错，而不是重新分配内存的原因是，假如buffer的实际指针被外部使用，这个时候就会产生严重不可控的后果。

<!-- 深拷贝函数`clone`与浅拷贝（拷贝构造函数）不同，它会创建一个完全独立的Tensor副本，包括复制底层的buffer数据。这在需要对Tensor进行修改而不影响原始Tensor的场景中非常有用。深拷贝通常会消耗更多的内存和计算资源，但能确保数据的独立性。 -->

对ir序列化的支持，序列化与反序列化函数允许将Tensor保存到文件或从文件加载。这对于模型权重的保存和加载非常重要。NNDeploy支持多种序列化格式，包括原生格式和与其他框架兼容的格式（如safetensors）。序列化时，会保存Tensor的描述信息（如形状、数据类型）和实际数据；反序列化时，则会根据这些信息重建Tensor。

由于Tensor在深度学习框架中被广泛使用，调试和可视化Tensor的内容非常重要。因此，Tensor类提供了打印函数，可以将Tensor的基本信息（如形状、数据类型）和实际数据值以可读的形式输出，方便开发者进行调试。

此外，Tensor类还提供了一系列帮助函数，包括：
- 数据访问函数：如`getData()`获取底层数据指针
- 属性查询函数：如`getShape()`、`getDataType()`、`getSize()`等
- 设备相关函数：如`getDevice()`、`getDeviceType()`等
- 内存管理函数：如`allocate()`、`deallocate()`等
- 数据操作函数：如`set()`设置所有元素为指定值

这些帮助函数使得Tensor的使用更加灵活和方便，能够满足各种深度学习场景的需求。


好了，有关我们 Tensor 的实现，我们就讲到这里了，同学们，拜拜。