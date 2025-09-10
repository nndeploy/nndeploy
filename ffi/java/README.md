# nndeploy Java FFI 实现

## 概述

本目录包含了nndeploy框架的Java Foreign Function Interface (FFI)实现，允许Java应用程序调用nndeploy的C++核心功能。实现基于JNI（Java Native Interface）技术，提供了完整的DAG图执行功能。

## 文件结构

```
ffi/java/
├── README.md                           # 本文档
├── config.cmake                        # CMake配置文件
├── nndeploy/                           # Java类文件目录
│   ├── GraphRunner.java               # 主要的图执行器类
│   ├── GraphRunnerResult.java         # 执行结果封装类
│   └── GraphRunnerExample.java        # 使用示例代码
└── jni/                               # JNI实现目录
    └── dag/                           # DAG模块JNI实现
        ├── graph_runner.h             # JNI头文件声明
        └── graph_runner.cc            # JNI实现代码
```

## 核心组件

### 1. GraphRunner类

`GraphRunner`是nndeploy Java API的主要入口类，提供了以下功能：

- **图执行**: 执行JSON格式定义的计算图
- **配置管理**: 支持性能分析、调试、并行类型等配置
- **资源管理**: 自动管理native资源，支持try-with-resources语法
- **参数设置**: 支持动态设置节点参数

#### 主要方法

```java
// 创建实例
GraphRunner runner = new GraphRunner();

// 配置选项
runner.setTimeProfile(true);        // 启用性能分析
runner.setDebug(false);             // 禁用调试模式
runner.setParallelType(GraphRunner.ParallelType.PIPELINE);

// 执行图计算
GraphRunnerResult result = runner.run(graphJson, "GraphName", "task_001");

// 设置节点参数
runner.setNodeValue("node_name", "param_key", "param_value");

// 释放资源
runner.close();
```

### 2. GraphRunnerResult类

封装图执行结果，包含：

- **状态信息**: 执行状态码和消息
- **性能数据**: 各节点执行时间统计
- **结果数据**: 图执行的输出结果

#### 主要方法

```java
// 检查执行状态
if (result.isSuccess()) {
    System.out.println("执行成功");
}

// 获取性能数据
float totalTime = result.getTotalTime();
float nodeTime = result.getNodeTime("node_name");

// 访问结果数据
int resultCount = result.getResultCount();
```

### 3. JNI实现层

JNI实现提供了Java和C++之间的桥接：

- **类型转换**: Java字符串与C++字符串的安全转换
- **内存管理**: 正确的native对象生命周期管理
- **异常处理**: 将C++异常转换为Java异常
- **性能优化**: 最小化JNI调用开销

## 使用示例

### 基本使用

```java
public class BasicExample {
    public static void main(String[] args) {
        String graphJson = """
        {
            "name": "SimpleGraph",
            "nodes": [
                {
                    "key": "Input",
                    "name": "input_node"
                },
                {
                    "key": "Process",
                    "name": "process_node"
                }
            ]
        }
        """;
        
        try (GraphRunner runner = new GraphRunner()) {
            runner.setTimeProfile(true);
            
            GraphRunnerResult result = runner.run(graphJson, "SimpleGraph", "task_001");
            
            if (result.isSuccess()) {
                System.out.println("执行成功，耗时: " + result.getTotalTime() + "ms");
            } else {
                System.err.println("执行失败: " + result.statusMessage);
            }
        }
    }
}
```

### 高级配置

```java
public class AdvancedExample {
    public static void main(String[] args) {
        try (GraphRunner runner = new GraphRunner()) {
            // 详细配置
            runner.setJsonFile(false);  // 输入是JSON字符串
            runner.setTimeProfile(true);
            runner.setDebug(true);
            runner.setParallelType(GraphRunner.ParallelType.TASK);
            runner.setLoopMaxFlag(true);
            
            // 设置节点参数
            runner.setNodeValue("model_node", "batch_size", "32");
            runner.setNodeValue("model_node", "device", "gpu");
            runner.setNodeValue("model_node", "precision", "fp16");
            
            // 执行图计算
            GraphRunnerResult result = runner.run(complexGraphJson, "ComplexGraph", "advanced_task");
            
            // 详细分析结果
            if (result.isSuccess()) {
                System.out.println("总执行时间: " + result.getTotalTime() + "ms");
                
                // 分析各节点性能
                result.timeProfilerMap.forEach((nodeName, time) -> 
                    System.out.println(nodeName + ": " + time + "ms")
                );
            }
        }
    }
}
```

### 批量处理

```java
public class BatchExample {
    public static void main(String[] args) {
        try (GraphRunner runner = new GraphRunner()) {
            runner.setParallelType(GraphRunner.ParallelType.PIPELINE);
            
            // 批量处理多个任务
            for (int i = 0; i < 10; i++) {
                String taskId = "batch_task_" + i;
                GraphRunnerResult result = runner.run(graphJson, "BatchGraph", taskId);
                
                if (result.isSuccess()) {
                    System.out.println("任务 " + taskId + " 完成，耗时: " + result.getTotalTime() + "ms");
                }
            }
        }
    }
}
```

## 配置选项详解

### 并行类型 (ParallelType)

- `NONE`: 无并行，顺序执行所有节点
- `SEQUENCE`: 顺序执行模式
- `PIPELINE`: 流水线并行，支持节点间并行执行
- `TASK`: 任务级并行，支持独立任务并行

### 调试和分析选项

- `setTimeProfile(boolean)`: 启用/禁用时间性能分析
- `setDebug(boolean)`: 启用/禁用调试模式，输出详细执行信息
- `setDump(boolean)`: 启用/禁用转储功能，输出图结构信息
- `setLoopMaxFlag(boolean)`: 启用/禁用循环最大标志

### 输入格式选项

- `setJsonFile(boolean)`: 
  - `true`: 输入参数是JSON文件路径
  - `false`: 输入参数是JSON字符串内容

## 构建和部署

### 依赖要求

- Java 8+
- CMake 3.12+
- nndeploy C++库
- JNI开发环境

### 构建步骤

1. 配置CMake构建：
```bash
mkdir build && cd build
cmake .. -DENABLE_NNDEPLOY_JAVA_FFI=ON
```

2. 编译native库：
```bash
make nndeploy_jni
```

3. 编译Java类：
```bash
javac -cp . com/nndeploy/dag/*.java
```

4. 运行示例：
```bash
java -Djava.library.path=./build/lib com.nndeploy.dag.GraphRunnerExample
```

### 部署注意事项

1. **库路径**: 确保`libnndeploy_jni.so`(Linux)或`nndeploy_jni.dll`(Windows)在Java库路径中
2. **依赖库**: 确保nndeploy核心库及其依赖可以被找到
3. **JVM设置**: 可能需要调整JVM堆内存大小以处理大型模型

## 性能优化建议

1. **复用GraphRunner实例**: 避免频繁创建和销毁GraphRunner实例
2. **批量处理**: 对于多个相似任务，使用同一个GraphRunner实例
3. **并行配置**: 根据硬件特性选择合适的并行类型
4. **内存管理**: 及时调用`close()`方法释放native资源

## 错误处理

### 常见错误类型

1. **UnsatisfiedLinkError**: native库加载失败
   - 检查库路径设置
   - 确认库文件存在且有执行权限

2. **RuntimeException**: GraphRunner创建失败
   - 检查nndeploy核心库是否正确初始化
   - 确认系统资源充足

3. **IllegalArgumentException**: 参数错误
   - 检查JSON格式是否正确
   - 确认必需参数不为空

4. **IllegalStateException**: GraphRunner已关闭
   - 确保在调用方法前GraphRunner未被关闭
   - 使用try-with-resources模式管理资源

### 调试技巧

1. 启用调试模式：`runner.setDebug(true)`
2. 启用转储功能：`runner.setDump(true)`
3. 检查执行结果状态和消息
4. 分析性能数据识别瓶颈

## 扩展开发

### 添加新的JNI接口

1. 在`graph_runner.h`中声明新的JNI函数
2. 在`graph_runner.cc`中实现JNI函数
3. 在Java类中添加对应的native方法声明
4. 更新CMake配置文件

### 自定义节点支持

当前实现主要支持GraphRunner执行预定义的图。要支持自定义节点：

1. 扩展JNI接口支持Node和Edge的直接操作
2. 实现Java端的Node和Edge包装类
3. 提供图构建API而不仅仅是JSON执行

## 许可证

本实现遵循nndeploy项目的Apache 2.0许可证。

## 贡献指南

欢迎提交bug报告、功能请求和代码贡献。请确保：

1. 遵循现有代码风格
2. 添加适当的测试用例
3. 更新相关文档
4. 确保JNI内存管理正确

## 联系方式

- 项目主页: https://github.com/nndeploy/nndeploy
- 问题反馈: https://github.com/nndeploy/nndeploy/issues
