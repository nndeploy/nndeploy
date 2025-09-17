# nndeploy 多语言FFI支持方案

## 概述

本文档详细规划了nndeploy框架的多语言Foreign Function Interface (FFI)支持方案，旨在将nndeploy的强大AI部署能力扩展到多种编程语言和平台，支持Android App、iOS App、Windows App、Mac App、后台程序等多种应用场景。

## 目标平台与语言支持

### 移动平台
- **Android**: Java/Kotlin + JNI
- **iOS**: Swift/Objective-C + C API

### 桌面平台  
- **Windows**: C#/.NET + P/Invoke
- **macOS**: Swift + C API
- **Linux**: Go + CGO, Python + ctypes

### Web平台
- **WebAssembly**: JavaScript + WASM
- **Node.js**: JavaScript + Native Addons

### 后端服务
- **Go**: CGO绑定
- **Rust**: FFI绑定
- **Java**: JNI绑定
- **C#**: P/Invoke绑定

## 技术架构设计

### 1. 核心C API层 (pure_c/)

#### 1.1 设计理念
- **最小化接口**: 只暴露核心功能，保持API简洁
- **C兼容性**: 纯C接口，避免C++特性，确保跨语言兼容
- **内存管理**: 明确的资源生命周期管理
- **错误处理**: 统一的错误码和状态返回机制
- **线程安全**: 支持多线程并发调用

#### 1.2 核心API设计

```c
// 基础框架管理
typedef struct nndeploy_context_t nndeploy_context_t;
typedef struct nndeploy_graph_t nndeploy_graph_t;
typedef struct nndeploy_tensor_t nndeploy_tensor_t;

// 错误码定义
typedef enum {
    NNDEPLOY_SUCCESS = 0,
    NNDEPLOY_ERROR_INVALID_PARAM = -1,
    NNDEPLOY_ERROR_OUT_OF_MEMORY = -2,
    NNDEPLOY_ERROR_RUNTIME_ERROR = -3,
    NNDEPLOY_ERROR_NOT_IMPLEMENTED = -4
} nndeploy_status_t;

// 框架初始化/销毁
NNDEPLOY_C_API nndeploy_status_t nndeploy_init();
NNDEPLOY_C_API nndeploy_status_t nndeploy_cleanup();
NNDEPLOY_C_API const char* nndeploy_get_version();

// 上下文管理
NNDEPLOY_C_API nndeploy_context_t* nndeploy_context_create();
NNDEPLOY_C_API nndeploy_status_t nndeploy_context_destroy(nndeploy_context_t* ctx);

// 图构建与执行
NNDEPLOY_C_API nndeploy_graph_t* nndeploy_graph_create_from_json(
    nndeploy_context_t* ctx, 
    const char* json_str
);
NNDEPLOY_C_API nndeploy_status_t nndeploy_graph_run(
    nndeploy_graph_t* graph,
    nndeploy_tensor_t** inputs,
    int input_count,
    nndeploy_tensor_t** outputs,
    int* output_count
);
NNDEPLOY_C_API nndeploy_status_t nndeploy_graph_destroy(nndeploy_graph_t* graph);

// 张量操作
NNDEPLOY_C_API nndeploy_tensor_t* nndeploy_tensor_create(
    const int* shape, 
    int ndim, 
    int dtype
);
NNDEPLOY_C_API nndeploy_status_t nndeploy_tensor_set_data(
    nndeploy_tensor_t* tensor, 
    void* data, 
    size_t size
);
NNDEPLOY_C_API void* nndeploy_tensor_get_data(nndeploy_tensor_t* tensor);
NNDEPLOY_C_API nndeploy_status_t nndeploy_tensor_destroy(nndeploy_tensor_t* tensor);
```

#### 1.3 实现文件结构
```
pure_c/
├── include/
│   └── nndeploy_c.h          # 主要C API头文件
├── src/
│   ├── nndeploy_c.cpp        # C API实现
│   ├── context.cpp           # 上下文管理
│   ├── graph.cpp             # 图操作封装
│   ├── tensor.cpp            # 张量操作封装
│   └── utils.cpp             # 工具函数
└── CMakeLists.txt            # 构建配置
```

### 2. JNI绑定层 (jni/)

#### 2.1 Android支持
- **目标**: 支持Android App开发，提供Java/Kotlin友好的API
- **技术栈**: JNI + NDK + Gradle

#### 2.2 Java API设计
```java
public class NNDeploy {
    // 单例模式
    public static NNDeploy getInstance();
    
    // 框架管理
    public boolean init();
    public void cleanup();
    public String getVersion();
    
    // 图执行
    public class Graph {
        public static Graph fromJson(String jsonStr);
        public Tensor[] run(Tensor[] inputs);
        public void destroy();
    }
    
    // 张量操作
    public class Tensor {
        public Tensor(int[] shape, DataType dtype);
        public void setData(float[] data);
        public float[] getData();
        public void destroy();
    }
}
```

#### 2.3 文件结构
```
jni/
├── android/
│   ├── gradle/              # Gradle构建配置
│   ├── src/main/java/       # Java源码
│   │   └── com/nndeploy/
│   │       ├── NNDeploy.java
│   │       ├── Graph.java
│   │       └── Tensor.java
│   └── src/main/cpp/        # JNI C++源码
│       ├── nndeploy_jni.cpp
│       └── CMakeLists.txt
└── java/
    └── # 纯Java版本(通过JNA调用)
```

### 3. Swift绑定层 (swift/)

#### 3.1 iOS/macOS支持
- **目标**: 支持iOS App和macOS App开发
- **技术栈**: Swift Package Manager + C API

#### 3.2 Swift API设计
```swift
public class NNDeploy {
    // 单例
    public static let shared = NNDeploy()
    
    // 框架管理
    public func initialize() -> Bool
    public func cleanup()
    public var version: String { get }
    
    // 图操作
    public class Graph {
        public init?(jsonString: String)
        public func run(inputs: [Tensor]) -> [Tensor]?
        deinit
    }
    
    // 张量操作  
    public class Tensor {
        public init(shape: [Int], dataType: DataType)
        public func setData<T>(_ data: [T])
        public func getData<T>() -> [T]?
        deinit
    }
}
```

#### 3.3 文件结构
```
swift/
├── Package.swift            # Swift Package配置
├── Sources/
│   └── NNDeploy/
│       ├── NNDeploy.swift   # 主要Swift API
│       ├── Graph.swift      # 图操作
│       ├── Tensor.swift     # 张量操作
│       └── Utils.swift      # 工具函数
├── Tests/
│   └── NNDeployTests/
└── README.md
```

### 4. C#绑定层 (csharp/)

#### 4.1 .NET支持
- **目标**: 支持Windows App、跨平台.NET应用
- **技术栈**: P/Invoke + NuGet

#### 4.2 C# API设计
```csharp
namespace NNDeploy
{
    public class NNDeployContext : IDisposable
    {
        // 单例
        public static NNDeployContext Instance { get; }
        
        // 框架管理
        public bool Initialize();
        public void Cleanup();
        public string Version { get; }
        
        // 图操作
        public class Graph : IDisposable
        {
            public static Graph FromJson(string jsonStr);
            public Tensor[] Run(Tensor[] inputs);
            public void Dispose();
        }
        
        // 张量操作
        public class Tensor : IDisposable
        {
            public Tensor(int[] shape, DataType dtype);
            public void SetData<T>(T[] data);
            public T[] GetData<T>();
            public void Dispose();
        }
    }
}
```

#### 4.3 文件结构
```
csharp/
├── NNDeploy.csproj         # .NET项目文件
├── src/
│   ├── NNDeployContext.cs  # 主要C# API
│   ├── Graph.cs            # 图操作
│   ├── Tensor.cs           # 张量操作
│   ├── Interop.cs          # P/Invoke声明
│   └── Enums.cs            # 枚举定义
├── tests/
│   └── NNDeployTests/
└── README.md
```

### 5. Go绑定层 (go/)

#### 5.1 Go支持
- **目标**: 支持Go后端服务开发
- **技术栈**: CGO + Go Modules

#### 5.2 Go API设计
```go
package nndeploy

// 框架管理
func Init() error
func Cleanup()
func GetVersion() string

// 上下文
type Context struct {
    handle unsafe.Pointer
}
func NewContext() (*Context, error)
func (ctx *Context) Close() error

// 图操作
type Graph struct {
    handle unsafe.Pointer
}
func (ctx *Context) NewGraphFromJSON(jsonStr string) (*Graph, error)
func (g *Graph) Run(inputs []*Tensor) ([]*Tensor, error)
func (g *Graph) Close() error

// 张量操作
type Tensor struct {
    handle unsafe.Pointer
}
func NewTensor(shape []int, dtype DataType) (*Tensor, error)
func (t *Tensor) SetData(data interface{}) error
func (t *Tensor) GetData() (interface{}, error)
func (t *Tensor) Close() error
```

#### 5.3 文件结构
```
go/
├── go.mod                  # Go模块定义
├── nndeploy.go            # 主要Go API
├── context.go             # 上下文管理
├── graph.go               # 图操作
├── tensor.go              # 张量操作
├── types.go               # 类型定义
├── examples/
│   └── simple_inference.go
└── README.md
```

## 实施路径规划

### 阶段一：基础设施建设 (1-2个月)

#### 1.1 核心C API开发 (优先级：最高)
- **任务**: 开发pure_c/目录下的核心C API
- **里程碑**:
  - 完成基础框架初始化/清理API
  - 实现图创建、执行、销毁API  
  - 完成张量创建、数据操作API
  - 添加错误处理和内存管理
- **验收标准**: C API能够成功执行基本的推理任务

#### 1.2 构建系统集成
- **任务**: 将C API集成到nndeploy主构建系统
- **里程碑**:
  - 修改CMakeLists.txt支持C API编译
  - 添加动态库/静态库构建选项
  - 支持跨平台编译(Linux/Windows/macOS)
- **验收标准**: 能够在各平台成功编译出C API库

### 阶段二：移动平台支持 (2-3个月)

#### 2.1 Android JNI绑定 (优先级：高)
- **任务**: 开发Android平台的Java/Kotlin API
- **里程碑**:
  - 完成JNI C++绑定层
  - 实现Java API封装
  - 集成Android NDK构建
  - 创建示例Android App
- **验收标准**: Android App能够成功调用nndeploy进行推理

#### 2.2 iOS Swift绑定 (优先级：高)
- **任务**: 开发iOS平台的Swift API
- **里程碑**:
  - 完成Swift Package封装
  - 实现Swift API绑定
  - 支持iOS/macOS双平台
  - 创建示例iOS App
- **验收标准**: iOS App能够成功调用nndeploy进行推理

### 阶段三：桌面平台支持 (2-3个月)

#### 3.1 Windows C#绑定 (优先级：中)
- **任务**: 开发Windows平台的.NET API
- **里程碑**:
  - 完成P/Invoke声明
  - 实现C# API封装
  - 支持.NET Framework和.NET Core
  - 创建示例WPF/WinUI应用
- **验收标准**: Windows桌面应用能够成功使用nndeploy

#### 3.2 macOS Swift支持增强
- **任务**: 增强macOS平台支持
- **里程碑**:
  - 优化macOS特定功能
  - 支持Metal GPU加速
  - 创建示例macOS App
- **验收标准**: macOS应用能够高效使用nndeploy

### 阶段四：服务端支持 (1-2个月)

#### 4.1 Go绑定 (优先级：中)
- **任务**: 开发Go语言后端服务支持
- **里程碑**:
  - 完成CGO绑定
  - 实现Go API封装
  - 添加并发安全支持
  - 创建HTTP服务示例
- **验收标准**: Go服务能够高并发处理推理请求

#### 4.2 其他语言支持 (优先级：低)
- **Rust绑定**: 通过rust-bindgen自动生成
- **Node.js绑定**: 通过node-ffi或N-API
- **Python ctypes**: 作为现有Python绑定的补充

### 阶段五：生态完善 (持续)

#### 5.1 文档和示例
- **任务**: 完善各语言的文档和示例
- **里程碑**:
  - 编写各语言API文档
  - 创建完整示例项目
  - 制作教程和最佳实践指南
- **验收标准**: 开发者能够快速上手各语言绑定

#### 5.2 CI/CD和测试
- **任务**: 建立完整的测试和发布流程
- **里程碑**:
  - 添加各平台自动化测试
  - 建立包发布流程(NuGet/Maven/CocoaPods等)
  - 集成性能基准测试
- **验收标准**: 所有绑定都有完整的测试覆盖

## 技术挑战与解决方案

### 1. 内存管理
**挑战**: 不同语言的内存管理模式差异巨大
**解决方案**: 
- C API层负责所有内存分配/释放
- 各语言绑定层只负责生命周期管理
- 使用RAII模式确保资源正确释放

### 2. 错误处理
**挑战**: 统一的错误处理机制
**解决方案**:
- C API返回统一错误码
- 各语言绑定转换为该语言的异常机制
- 提供详细的错误信息和堆栈跟踪

### 3. 线程安全
**挑战**: 多线程环境下的安全性
**解决方案**:
- C API层实现线程安全
- 使用互斥锁保护共享资源
- 各语言绑定继承线程安全特性

### 4. 性能优化
**挑战**: FFI调用的性能开销
**解决方案**:
- 最小化跨语言调用次数
- 批量操作减少调用开销
- 使用零拷贝技术传递大数据

## 发布策略

### 包管理平台
- **Android**: Maven Central / JCenter
- **iOS**: CocoaPods / Swift Package Manager
- **Windows**: NuGet
- **Go**: Go Modules
- **Node.js**: NPM
- **Python**: PyPI (补充ctypes版本)

### 版本管理
- 采用语义化版本控制(SemVer)
- 各语言绑定版本与核心框架版本保持同步
- 提供长期支持(LTS)版本

### 文档和支持
- 各语言独立的API文档
- 统一的示例代码仓库
- 社区支持和问题反馈机制

## 成功指标

### 技术指标
- [ ] 支持5种以上主流编程语言
- [ ] 覆盖移动、桌面、服务端三大平台
- [ ] FFI调用性能损失 < 5%
- [ ] 内存泄漏率 < 0.01%

### 生态指标  
- [ ] 各语言都有完整示例项目
- [ ] API文档覆盖率 > 95%
- [ ] 单元测试覆盖率 > 90%
- [ ] 社区贡献者 > 10人

### 商业指标
- [ ] 月活跃开发者 > 1000人
- [ ] 生产环境部署 > 100个项目
- [ ] 社区Star数 > 10000

## 总结

通过分阶段实施这个多语言FFI支持方案，nndeploy将能够：

1. **扩大用户群体**: 支持更多编程语言的开发者
2. **覆盖全平台**: 从移动端到服务端的完整覆盖
3. **降低接入门槛**: 各语言原生API，易于集成
4. **保持高性能**: 最小化FFI开销，保持推理性能
5. **建立生态**: 形成完整的多语言开发者生态

这个方案将使nndeploy成为真正的跨语言、跨平台的AI部署框架，大大扩展其应用场景和市场影响力。
