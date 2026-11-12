---
name: feature_edge_default_value
title: 边 Edge 默认值支持
description: 为DAG边添加默认值机制，支持常量、Tensor、节点输出和表达式等多种默认值类型，提高工作流灵活性
category: [feature]
difficulty: medium
priority: P3
status: planned
version: 1.0.0
tags: [dag, edge, default-value, data-packet, pipeline]
estimated_time: 8h
files_affected: [framework/include/nndeploy/dag/edge.h, framework/include/nndeploy/dag/edge/data_packet.h, framework/source/nndeploy/dag/edge.cc, framework/source/nndeploy/dag/edge/data_packet.cc, framework/source/nndeploy/dag/edge/pipeline_edge.cc]
---

# Feature: 边 Edge 默认值支持

## 1. 背景（是什么 && 为什么）

### 现状分析
- 当前 nndeploy 的 DAG Edge（边）主要用作数据传递通道，负责从一个节点向另一个节点传递数据
- Edge 没有默认值机制，当上游节点没有输出数据时，下游节点会阻塞或获得空数据
- 在实际应用中，很多场景需要为输入提供默认值：
  - 可选参数：某些参数不是必须的，需要默认值
  - 初始状态：工作流启动时某些输入可能还没有就绪
  - 容错机制：上游节点失败时，下游节点可以使用默认值继续运行
  - 配置化：将默认值作为工作流配置的一部分

### 设计问题
- **具体的技术问题**: 缺少默认值机制，需要每个节点自己处理空输入
- **架构层面的不足**: 边的设计过于简单，缺少高级特性
- **用户体验的缺陷**:
  - 工作流配置不够灵活
  - 需要在节点代码中处理默认值逻辑
  - 难以通过配置方式定义默认行为

## 2. 目标（想做成什么样子）

### 核心目标
- 为 Edge 添加默认值支持
- 当上游节点没有输出时，Edge 返回配置的默认值
- 支持多种数据类型的默认值（Tensor、Buffer、基础类型等）
- 提供配置接口和 API 接口

### 预期效果
- **功能层面的改进**:
  - 边可以配置默认值
  - 工作流更加灵活和容错
  - 支持可选输入的场景
- **性能层面的提升**: 无明显性能影响
- **用户体验的优化**:
  - 通过 JSON 配置设置默认值
  - 减少节点中的默认值处理代码
  - 工作流配置更直观

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要修改的文件
- `framework/include/nndeploy/dag/edge.h` - 添加默认值相关接口
- `framework/include/nndeploy/dag/edge/data_packet.h` - 支持默认值的数据包
- `framework/source/nndeploy/dag/edge.cc` - 实现默认值逻辑
- `framework/source/nndeploy/dag/edge/data_packet.cc` - 实现默认值数据包
- `framework/source/nndeploy/dag/edge/pipeline_edge.cc` - Pipeline Edge 默认值支持
- `framework/include/nndeploy/dag/node.h` - Node 可能需要适配默认值处理

### 需要新增的文件
- `framework/include/nndeploy/dag/edge/default_value.h` - 默认值类型定义
- `framework/source/nndeploy/dag/edge/default_value.cc` - 默认值实现
- `framework/include/nndeploy/dag/edge/data_packet_with_default.h` - 带默认值的数据包

### 不能修改的文件
- 现有插件的核心实现 - 保持向后兼容
- 公共 API 中不兼容的部分 - 通过新增接口实现

### 影响范围
- 所有 Edge 类型
- 数据包传递机制
- JSON workflow 配置格式
- 节点的输入处理逻辑

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**: Edge 不支持默认值，空输入需要每个节点自己处理
- **新方案**: Edge 支持配置默认值，没有输入时自动返回默认值
- **核心变化**: 在 Edge 和 DataPacket 中添加默认值管理

### 架构/接口设计

#### 1. 默认值类型定义
```cpp
// framework/include/nndeploy/dag/edge/default_value.h

namespace nndeploy {
namespace dag {

// 默认值类型
enum class DefaultValueSourceType {
  kNone,              // 无默认值
  kConstant,          // 常量默认值
  kExpression,        // 表达式默认值（如: "previous_output * 0.5"）
  kNodeOutput,        // 从其他节点输出获取默认值
};

// 默认值基类
class NNDEPLOY_CC_API DefaultValueBase {
 public:
  virtual ~DefaultValueBase() = default;

  virtual DefaultValueSourceType getType() const = 0;
  virtual bool isValid() const = 0;
  virtual void reset() = 0;
};

// 常量默认值
template<typename T>
class NNDEPLOY_CC_API ConstantDefaultValue : public DefaultValueBase {
 public:
  explicit ConstantDefaultValue(const T& value) : value_(value) {}

  DefaultValueSourceType getType() const override {
    return DefaultValueSourceType::kConstant;
  }

  bool isValid() const override { return true; }

  void reset() override {}

  const T& getValue() const { return value_; }

 private:
  T value_;
};

// Tensor 默认值
class NNDEPLOY_CC_API TensorDefaultValue : public DefaultValueBase {
 public:
  TensorDefaultValue(const device::Device* device,
                  const device::TensorDesc& desc,
                  float fill_value = 0.0f);

  DefaultValueSourceType getType() const override {
    return DefaultValueSourceType::kConstant;
  }

  bool isValid() const override { return tensor_ != nullptr; }

  void reset() override {}

  device::Tensor* getTensor() { return tensor_; }

 private:
  std::unique_ptr<device::Tensor> tensor_;
};

// 表达式默认值（预留，支持简单表达式）
class NNDEPLOY_CC_API ExpressionDefaultValue : public DefaultValueBase {
 public:
  explicit ExpressionDefaultValue(const std::string& expression);

  DefaultValueSourceType getType() const override {
    return DefaultValueSourceType::kExpression;
  }

  bool isValid() const override { return !expression_.empty(); }

  void reset() override {}

  const std::string& getExpression() const { return expression_; }

  // 预留：表达式求值接口
  // std::any evaluate(const std::map<std::string, std::any>& context);

 private:
  std::string expression_;
};

// 其他节点输出作为默认值
class NNDEPLOY_CC_API NodeOutputDefaultValue : public DefaultValueBase {
 public:
  NodeOutputDefaultValue(const std::string& node_name, int output_index = 0);

  DefaultValueSourceType getType() const override {
    return DefaultValueSourceType::kNodeOutput;
  }

  bool isValid() const override { return !node_name_.empty(); }

  void reset() override {}

  const std::string& getNodeName() const { return node_name_; }
  int getOutputIndex() const { return output_index_; }

 private:
  std::string node_name_;
  int output_index_;
};

// 默认值包装器
class NNDEPLOY_CC_API DefaultValue {
 public:
  DefaultValue();  // 无默认值
  explicit DefaultValue(std::shared_ptr<DefaultValueBase> impl);

  bool hasValue() const { return impl_ != nullptr; }
  DefaultValueSourceType getType() const;
  bool isValid() const;

  // 获取不同类型的值
  template<typename T>
  T getAs() const;

  // Tensor 专用
  device::Tensor* getTensor() const;

 private:
  std::shared_ptr<DefaultValueBase> impl_;
};

}  // namespace dag
}  // namespace nndeploy
```

#### 2. Edge 扩展
```cpp
// framework/include/nndeploy/dag/edge.h (扩展部分)

class NNDEPLOY_CC_API Edge {
 public:
  // ... 现有接口 ...

  // 默认值相关接口
  void setDefaultValue(const DefaultValue& default_value);
  void setDefaultValue(float value);
  void setDefaultValue(int value);
  void setDefaultValue(const std::string& value);
  void setTensorDefaultValue(const device::Device* device,
                         const device::TensorDesc& desc,
                         float fill_value = 0.0f);

  void setDefaultValueFromNode(const std::string& node_name, int output_index = 0);
  void setDefaultValueExpression(const std::string& expression);

  const DefaultValue& getDefaultValue() const { return default_value_; }
  bool hasDefaultValue() const { return default_value_.hasValue(); }

  // 检查是否有实际数据（非默认值）
  bool hasActualData(const Node* node) const;

  // 获取数据（如果没有实际数据则返回默认值）
  device::Buffer* getBufferWithDefault(const Node* node);
  device::Tensor* getTensorWithDefault(const Node* node);
  base::Mat* getMatWithDefault(const Node* node);

 private:
  DefaultValue default_value_;
};
```

#### 3. DataPacket 扩展
```cpp
// framework/include/nndeploy/dag/edge/data_packet.h (扩展部分)

class NNDEPLOY_CC_API DataPacket {
 public:
  // ... 现有接口 ...

  // 默认值支持
  void setDefaultValue(const DefaultValue& default_value);
  const DefaultValue& getDefaultValue() const { return default_value_; }
  bool hasDefaultValue() const { return default_value_.hasValue(); }

  // 获取数据（带默认值）
  device::Buffer* getBufferWithDefault() const;
  device::Tensor* getTensorWithDefault() const;

 protected:
  DefaultValue default_value_;
};
```

#### 4. PipelineEdge 默认值实现
```cpp
// framework/source/nndeploy/dag/edge/pipeline_edge.cc (扩展部分)

class NNDEPLOY_CC_API PipelineEdge : public Edge {
 public:
  // ... 现有实现 ...

  device::Buffer* getBuffer(const Node* node) override {
    std::unique_lock<std::mutex> lock(mutex_);

    // 检查是否有实际数据
    if (!written_ && hasDefaultValue()) {
      // 返回默认值
      return createDefaultBuffer();
    }

    cv_.wait(lock, [this] { return written_; });
    return abstact_edge_->getBuffer(node);
  }

 private:
  device::Buffer* createDefaultBuffer() {
    const DefaultValue& default_value = getDefaultValue();
    if (!default_value.hasValue()) {
      return nullptr;
    }

    // 根据默认值类型创建 Buffer
    switch (default_value.getType()) {
      case DefaultValueSourceType::kConstant:
        return createConstantBuffer();
      case DefaultValueSourceType::kNodeOutput:
        return createNodeOutputBuffer();
      case DefaultValueSourceType::kExpression:
        return createExpressionBuffer();
      default:
        return nullptr;
    }
  }
};
```

#### 5. JSON 配置支持
```json
{
  "edges": [
    {
      "source": "preprocess_node",
      "source_port": 0,
      "target": "infer_node",
      "target_port": 0,
      "default_value": {
        "type": "tensor",
        "device": "cpu",
        "data_type": "float",
        "shape": [1, 3, 224, 224],
        "fill_value": 0.0
      }
    },
    {
      "source": "config_node",
      "source_port": 0,
      "target": "detect_node",
      "target_port": 1,
      "default_value": {
        "type": "constant",
        "value": 0.5
      }
    },
    {
      "source": "optional_input_node",
      "source_port": 0,
      "target": "process_node",
      "target_port": 0,
      "default_value": {
        "type": "node_output",
        "node_name": "fallback_node",
        "output_index": 0
      }
    }
  ]
}
```

### 核心操作流程

```
数据获取流程（带默认值）:

1. 节点调用 edge->getBuffer(node)

2. Edge 检查状态:
   a. 如果有上游节点的实际数据:
      -> 返回实际数据
   b. 如果没有实际数据且有默认值:
      -> 返回默认值
   c. 如果没有实际数据也没有默认值:
      -> 返回 nullptr 或阻塞（原有行为）

3. 默认值创建:
   a. 常量默认值: 创建包含常量的 Buffer
   b. Tensor 默认值: 创建指定形状和填充值的 Tensor
   c. 节点输出默认值: 从指定节点获取输出
   d. 表达式默认值: 求值后创建 Buffer

4. 节点处理数据
   -> 无论数据来自上游还是默认值，处理逻辑一致
```

### 技术细节
- 默认值是 Edge 的属性，不是 Node 的属性
- 默认值类型：无、常量、Tensor、节点输出、表达式
- 默认值的生命周期与 Edge 相同
- 支持 Pipeline 并行模式下的默认值
- 默认值可以序列化到 JSON
- 支持动态更新默认值

## 5. 实施步骤

### Step 1: 定义默认值数据结构
- 创建 `default_value.h` 和 `default_value.cc`
- 定义 `DefaultValueBase` 基类
- 实现各种默认值类型
- 涉及文件: 新增文件

### Step 2: 扩展 Edge 基类
- 修改 `framework/include/nndeploy/dag/edge.h`
- 添加默认值相关接口
- 修改 `framework/source/nndeploy/dag/edge.cc`
- 实现默认值逻辑
- 涉及文件: `framework/include/nndeploy/dag/edge.h`, `framework/source/nndeploy/dag/edge.cc`

### Step 3: 扩展 DataPacket
- 修改 `framework/include/nndeploy/dag/edge/data_packet.h`
- 添加默认值支持
- 修改 `framework/source/nndeploy/dag/edge/data_packet.cc`
- 实现 getBufferWithDefault 等方法
- 涉及文件: `framework/include/nndeploy/dag/edge/data_packet.h`, `framework/source/nndeploy/dag/edge/data_packet.cc`

### Step 4: 实现 PipelineEdge 默认值
- 修改 `framework/source/nndeploy/dag/edge/pipeline_edge.cc`
- 实现带默认值的数据获取
- 处理条件变量等待
- 涉及文件: `framework/source/nndeploy/dag/edge/pipeline_edge.cc`

### Step 5: JSON 配置支持
- 扩展 JSON 解析器
- 支持默认值配置
- 涉及文件: `framework/source/nndeploy/dag/graph.cc` 或相关 JSON 解析文件

### Step 6: 添加单元测试
- 测试常量默认值
- 测试 Tensor 默认值
- 测试节点输出默认值
- 测试无默认值的原有行为
- 测试 Pipeline 并行模式
- 涉及文件: `test/dag/edge/`

### Step 7: 集成测试和示例
- 创建使用默认值的示例工作流
- 验证实际应用场景
- 涉及文件: `demo/` 或 `example/`

### 兼容性与迁移
- **向后兼容策略**: 默认值为空时保持原有行为
- **迁移路径**: 现有工作流无需修改，默认值为空
- **过渡期安排**:
  - 阶段1: 基础设施完成
  - 阶段2: 支持常量和 Tensor 默认值
  - 阶段3: 支持节点输出和表达式默认值

## 6. 验收标准

### 功能测试
- **测试用例 1**: 常量默认值
  - 创建 Edge 并设置常量默认值
  - 上游节点无输出
  - 验证下游节点收到默认值
- **测试用例 2**: Tensor 默认值
  - 创建 Edge 并设置 Tensor 默认值
  - 验证 Tensor 形状和值正确
- **测试用例 3**: 节点输出默认值
  - 创建 Edge 并设置从其他节点获取默认值
  - 验证正确获取指定节点的输出
- **测试用例 4**: 有实际数据时忽略默认值
  - 上游节点有输出
  - Edge 有默认值
  - 验证返回实际数据而非默认值
- **测试用例 5**: Pipeline 并行模式
  - 在 Pipeline Edge 上设置默认值
  - 验证消费者能获取默认值
  - 验证没有死锁
- **测试用例 6**: JSON 配置
  - 通过 JSON 配置默认值
  - 加载工作流
  - 验证默认值正确设置
- **测试用例 7**: 无默认值（原有行为）
  - 不设置默认值
  - 验证原有行为不受影响

### 代码质量
- 代码通过编译
- 代码覆盖率 > 85%
- 符合现有代码规范
- API 文档完整

### 回归测试
- 现有 DAG 测试用例全部通过
- 现有 Edge 测试全部通过
- 工作流执行不受影响

### 性能与可维护性
- 默认值开销 < 5% CPU
- 内存开销可预测
- 代码结构清晰

### 文档与示例
- API 文档完整
- 提供配置示例
- 提供使用指南

## 7. 其他说明

### 相关资源
- 现有 Edge 设计
- 现有 DataPacket 设计
- JSON workflow 配置格式

### 风险与应对
- **潜在风险**: 默认值可能与实际数据类型不匹配
  - 应对措施: 添加类型检查，不匹配时返回错误
- **潜在风险**: 节点输出默认值可能导致循环依赖
  - 应对措施: 检测并拒绝循环依赖
- **潜在风险**: Pipeline 并行模式下默认值可能影响同步
  - 应对措施: 默认值设置时立即通知消费者

### 依赖关系
- **依赖**: 无
- **被依赖**: 可选输入节点、容错工作流、配置化工作流

### 扩展性
- 支持默认值的热更新
- 支持动态默认值（从配置文件加载）
- 支持默认值优先级（多个源）
- 支持默认值的版本管理

### 未来计划
- 支持表达式默认值的求值
- 支持默认值的条件判断
- 支持默认值的调试和监控
- 支持默认值的模板化（基于输入参数）
