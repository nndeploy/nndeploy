---
name: feature_reducer
title: 状态更新策略（Reducer 模式）
description: 支持灵活的DAG边数据更新策略，提供overwrite、append_list、add_messages、merge_map等多种内置reducer和自定义reducer函数
category: [feature]
difficulty: easy
priority: P3
status: planned
version: 1.0.0
tags: [dag, edge, reducer, state, data-merge]
estimated_time: 4h
files_affected: [framework/include/nndeploy/dag/edge.h, framework/source/nndeploy/dag/edge.cc, framework/include/nndeploy/dag/reducer.h, framework/source/nndeploy/dag/reducer.cc]
---

# Feature: 状态更新策略（Reducer 模式）

## 1. 背景（是什么 && 为什么）

### 现状分析
- **当前状态**：nndeploy DAG 通过 Edge 传输数据，数据更新是简单的覆盖模式
- **存在问题**：缺少灵活的状态更新策略，难以实现列表追加、消息累积等场景
- **需求痛点**：多轮对话需要消息列表累积，数组数据需要灵活的合并策略

### 设计问题
- Edge 只支持单一的数据传递方式
- 没有状态冲突解决机制
- 不支持自定义更新函数
- 多输入节点的数据合并策略不灵活

## 2. 目标（想做成什么样子）

### 核心目标
- **多种更新策略**：支持 overwrite（覆盖）、append_list（列表追加）、add_messages（消息追加）等策略
- **自定义 Reducer**：支持用户自定义更新函数
- **冲突解决**：多输入时的数据合并策略
- **声明式配置**：通过配置声明更新策略

### 预期效果
- Edge 支持配置 reducer 策略
- 内置常用 reducer 类型
- 支持自定义 reducer 函数
- 状态更新可预测

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要修改的文件
- `framework/include/nndeploy/dag/edge.h` - 添加 reducer 支持
- `framework/source/nndeploy/dag/edge.cc` - 实现 reducer 逻辑

### 需要新增的文件
- `framework/include/nndeploy/dag/reducer.h` - Reducer 基类和内置实现
- `framework/source/nndeploy/dag/reducer.cc` - Reducer 实现

### 不能修改的文件
- 现有的数据传输接口保持兼容
- Device、Tensor 等底层模块不做修改

### 影响范围
- 所有 Edge 的数据传递行为可能变化
- 需要保证向后兼容性

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：Edge 数据传递是简单的覆盖（新值替换旧值）
- **新方案**：支持多种 Reducer 策略，灵活控制数据更新
- **核心变化**：新增 Reducer 系统，Edge 集成 Reducer

### 架构/接口设计

#### 类型定义
```cpp
// 内置 Reducer 类型
enum ReducerType {
    kReducerOverwrite,       // 直接覆盖（默认，向后兼容）
    kReducerAppendList,      // 列表追加
    kReducerAddMessages,     // 消息列表追加（带去重）
    kReducerMergeMap,        // Map 合并
    kReducerUnionSet,        // Set 并集
    kReducerSum,             // 数值求和
    kReducerMax,             // 取最大值
    kReducerMin,             // 取最小值
    kReducerAvg,             // 取平均值
    kReducerCustom           // 自定义函数
};

// Reducer 基类
class Reducer {
public:
    virtual ~Reducer() = default;

    // 应用 reducer
    virtual nlohmann::json apply(const nlohmann::json& current,
                                   const nlohmann::json& incoming) = 0;

    // 获取 Reducer 类型
    virtual ReducerType getType() const = 0;

    // 获取 Reducer 名称
    virtual std::string getName() const = 0;
};

// 内置 Reducer 实现
class OverwriteReducer : public Reducer {
public:
    virtual nlohmann::json apply(const nlohmann::json& current,
                                  const nlohmann::json& incoming) override {
        return incoming;  // 直接返回新值
    }

    virtual ReducerType getType() const override { return kReducerOverwrite; }
    virtual std::string getName() const override { return "overwrite"; }
};

class AppendListReducer : public Reducer {
public:
    virtual nlohmann::json apply(const nlohmann::json& current,
                                  const nlohmann::json& incoming) override {
        nlohmann::json result = current.is_array() ? current : nlohmann::json::array();
        if (incoming.is_array()) {
            result.insert(result.end(), incoming.begin(), incoming.end());
        } else {
            result.push_back(incoming);
        }
        return result;
    }

    virtual ReducerType getType() const override { return kReducerAppendList; }
    virtual std::string getName() const override { return "append_list"; }
};

class AddMessagesReducer : public Reducer {
public:
    virtual nlohmann::json apply(const nlohmann::json& current,
                                  const nlohmann::json& incoming) override {
        nlohmann::json result = current.is_array() ? current : nlohmann::json::array();

        // 消息去重：根据 message_id 或 role+content
        if (incoming.is_array()) {
            for (const auto& msg : incoming) {
                if (!isDuplicate(result, msg)) {
                    result.push_back(msg);
                }
            }
        }

        return result;
    }

    virtual ReducerType getType() const override { return kReducerAddMessages; }
    virtual std::string getName() const override { return "add_messages"; }

private:
    bool isDuplicate(const nlohmann::json& messages, const nlohmann::json& msg) {
        std::string id = msg.value("message_id", "");
        if (!id.empty()) {
            for (const auto& m : messages) {
                if (m.value("message_id", "") == id) {
                    return true;
                }
            }
        }
        return false;
    }
};

class MergeMapReducer : public Reducer {
public:
    virtual nlohmann::json apply(const nlohmann::json& current,
                                  const nlohmann::json& incoming) override {
        nlohmann::json result = current.is_object() ? current : nlohmann::json::object();
        if (incoming.is_object()) {
            for (auto it = incoming.begin(); it != incoming.end(); ++it) {
                result[it.key()] = it.value();
            }
        }
        return result;
    }

    virtual ReducerType getType() const override { return kReducerMergeMap; }
    virtual std::string getName() const override { return "merge_map"; }
};

// 数值求和 Reducer
class SumReducer : public Reducer {
public:
    virtual nlohmann::json apply(const nlohmann::json& current,
                                  const nlohmann::json& incoming) override {
        if (current.is_number() && incoming.is_number()) {
            return current.get<double>() + incoming.get<double>();
        }
        return incoming;  // 非数值时返回 incoming
    }

    virtual ReducerType getType() const override { return kReducerSum; }
    virtual std::string getName() const override { return "sum";
}};

// 数值平均值 Reducer（需额外维护计数）
class AvgReducer : public Reducer {
public:
    AvgReducer() : count_(0), sum_(0.0) {}

    virtual nlohmann::json apply(const nlohmann::json& current,
                                  const nlohmann::json& incoming) override {
        if (incoming.is_number()) {
            sum_ += incoming.get<double>();
            count_++;
            return sum_ / count_;
        }
        return incoming;
    }

    virtual ReducerType getType() const override { return kReducerAvg; }
    virtual std::string getName() const override { return "avg"; }

private:
    double sum_;
    size_t count_;
};

// 取最大值 Reducer
class MaxReducer : public Reducer {
public:
    virtual nlohmann::json apply(const nlohmann::json& current,
                                  const nlohmann::json& incoming) override {
        if (!current.is_number()) return incoming;
        if (!incoming.is_number()) return current;
        return std::max(current.get<double>(), incoming.get<double>());
    }

    virtual ReducerType getType() const override { return kReducerMax; }
    virtual std::string getName() const override { return "max";
}};

// 取最小值 Reducer
class MinReducer : public Reducer {
public:
    virtual nlohmann::json apply(const nlohmann::json& current,
                                  const nlohmann::json& incoming) override {
        if (!current.is_number()) return incoming;
        if (!incoming.is_number()) return current;
        return std::min(current.get<double>(), incoming.get<double>());
    }

    virtual ReducerType getType() const override { return kReducerMin; }
    virtual std::string getName() const override { return "min";
}};

class CustomReducer : public Reducer {
public:
    // 自定义 Reducer 函数类型（性能考虑，避免频繁 std::function 调用）
    using ReducerFunc = std::function<nlohmann::json(
        const nlohmann::json&, const nlohmann::json&)>;

    CustomReducer(const std::string& name, ReducerFunc func,
                  const std::string& description = "")
        : name_(name), func_(func), description_(description) {}

    virtual nlohmann::json apply(const nlohmann::json& current,
                                  const nlohmann::json& incoming) override {
        if (!func_) {
            // 无自定义函数时返回 incoming（默认覆盖行为）
            return incoming;
        }
        try {
            return func_(current, incoming);
        } catch (const std::exception& e) {
            // 异常时返回 incoming，记录错误
            // LOG_ERROR << "CustomReducer " << name_ << " failed: " << e.what();
            return incoming;
        }
    }

    virtual ReducerType getType() const override { return kReducerCustom; }
    virtual std::string getName() const override { return name_; }
    virtual std::string getDescription() const override { return description_; }

private:
    std::string name_;
    ReducerFunc func_;
    std::string description_;
};
```

#### API 设计
```cpp
// Edge 扩展
class Edge {
public:
    // 设置 Reducer
    void setReducer(std::shared_ptr<Reducer> reducer);

    // 获取 Reducer
    std::shared_ptr<Reducer> getReducer() const;

    // 设置内置 Reducer
    void setReducerType(ReducerType type);

    // 应用 Reducer 更新数据
    nlohmann::json applyReducer(const nlohmann::json& incoming);

protected:
    std::shared_ptr<Reducer> reducer_;
};

// Reducer 工厂
class ReducerFactory {
public:
    // 创建内置 Reducer
    static std::shared_ptr<Reducer> create(ReducerType type);

    // 从 JSON 配置创建
    static std::shared_ptr<Reducer> create(const nlohmann::json& config);

    // 注册自定义 Reducer
    static void registerReducer(const std::string& name,
                                  std::function<std::shared_ptr<Reducer>()> creator);

    // 获取所有 Reducer 类型
    static std::vector<std::string> getAvailableTypes();

private:
    static std::map<std::string, std::function<std::shared_ptr<Reducer>()>> creators_;
    static std::mutex mutex_;
};

// 全局设置（可选）
namespace nndeploy {
namespace dag {

// 设置 Edge 的默认 Reducer 类型
void setDefaultReducerType(ReducerType type);

// 获取默认 Reducer 类型
ReducerType getDefaultReducerType();

}} // namespace nndeploy::dag
```

### 核心操作流程

#### 数据更新流程
```
┌─────────────────┐
│  Node A         │
│  产生数据       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Edge           │
│  applyReducer()  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reducer        │
│  apply()        │
│  应用更新策略    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  更新后的数据    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Node B         │
│  接收数据       │
└─────────────────┘
```

#### 多输入合并流程
```
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Node 1  │   │ Node 2  │   │ Node 3  │
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
     └──────────┬──┴─────────────┘
                ▼
        ┌───────────────┐
        │  Reducer      │
        │  apply() x N  │
        │  合并所有输入 │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │  Node 4      │
        └───────────────┘
```

### 技术细节
- 使用 nlohmann::json 作为数据格式
- Reducer 应用是幂等的（多次应用结果一致）
- 支持链式 Reducer（先应用 Reducer A，再应用 Reducer B）
- 类型检查和错误处理
- 避免频繁 std::function 调用的性能损失

## 5. 实施步骤

### Step 1: 定义 Reducer 基类和类型
- 定义 ReducerType 枚举
- 定义 Reducer 基类接口
- 涉及文件：`framework/include/nndeploy/dag/reducer.h`

### Step 2: 实现内置 Reducer
- 实现 OverwriteReducer
- 实现 AppendListReducer
- 实现 AddMessagesReducer
- 实现 MergeMapReducer
- 实现其他数值类 Reducer
- 涉及文件：`framework/include/nndeploy/dag/reducer.h`, `framework/source/nndeploy/dag/reducer.cc`

### Step 3: 实现 ReducerFactory
- 实现创建方法
- 实现注册机制
- 实现配置解析
- 涉及文件：`framework/include/nndeploy/dag/reducer.h`, `framework/source/nndeploy/dag/reducer.cc`

### Step 4: 扩展 Edge 类
- 添加 Reducer 成员
- 实现 setReducer() 和 getReducer()
- 实现 applyReducer() 方法
- 涉及文件：`framework/include/nndeploy/dag/edge.h`, `framework/source/nndeploy/dag/edge.cc`

### Step 5: 测试和文档
- 编写单元测试
- 编写使用示例
- 编写自定义 Reducer 示例
- 涉及文件：`test/`, `docs/`

### 兼容性与迁移
- 默认使用 OverwriteReducer，保持向后兼容
- 不设置 Reducer 时行为不变
- 现有代码无需修改

## 6. 验收标准

### 功能测试
- **测试用例 1**：OverwriteReducer 正确覆盖数据
- **测试用例 2**：AppendListReducer 正确追加列表
- **测试用例 3**：AddMessagesReducer 正确去重追加消息
- **测试用例 4**：MergeMapReducer 正确合并对象
- **测试用例 5**：数值 Reducer（Sum/Max/Min）正确计算
- **测试用例 6**：CustomReducer 正确执行自定义逻辑
- **测试用例 7**：多输入 Reducer 正确合并数据
- **测试用例 8**：不设置 Reducer 时默认行为不变

### 代码质量
- 通过 GoogleTest 单元测试
- 代码覆盖率 > 80%
- 符合项目代码规范
- 头文件注释完善

### 回归测试
- 现有 DAG 执行功能正常运行
- 现有数据传输行为不变（默认情况）
- 各种 Executor 仍能正常工作

### 性能与可维护性
- Reducer 执行时间 < 1ms
- 不增加数据传输延迟
- 代码结构清晰，易于扩展新的 Reducer

### 文档与示例
- API 文档完整
- 提供使用示例
- 提供自定义 Reducer 示例

## 7. 其他说明

### 相关资源
- Redux Reducer 概念
- LangGraph state reducer 文档

### 风险与应对
- **风险**：Reducer 应用顺序影响结果
  - **应对**：文档明确说明执行顺序，提供确定性保证
- **风险**：类型不匹配导致错误
  - **应对**：添加类型检查，提供友好的错误信息
- **风险**：自定义 Reducer 质量无法保证
  - **应对**：提供最佳实践文档，内置常用 Reducer

### 依赖关系
- 依赖：无（独立功能）
- 被依赖：消息历史管理

### 扩展方向
- 支持链式 Reducer
- 支持 Reducer 组合（先 A 后 B）
- 支持条件 Reducer（根据条件选择策略）
- 支持异步 Reducer
