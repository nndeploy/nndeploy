---
name: feature_context_management
title: 执行上下文（Context）管理
description: 提供统一的上下文管理机制，支持执行栈追踪、参数传递、线程安全访问和嵌套上下文
category: [feature]
difficulty: easy
priority: P2
status: planned
version: 1.0.0
tags: [dag, context, executor, thread-safe, tracing]
estimated_time: 4h
files_affected: [framework/include/nndeploy/dag/context.h, framework/source/nndeploy/dag/context.cc, framework/include/nndeploy/dag/executor.h, framework/include/nndeploy/dag/graph_runner.h]
---

# Feature: 执行上下文（Context）管理

## 1. 背景（是什么 && 为什么）

### 现状分析
- **当前状态**：nndeploy DAG 执行过程中缺少统一的上下文管理
- **存在问题**：无法追踪执行栈、传递运行时参数、管理调用链路
- **需求痛点**：需要统一的上下文传递机制，支持日志追踪、性能分析、跨层数据传递

### 设计问题
- 没有统一的 Context 类
- 无法传递运行时配置
- 调用链路追踪困难
- 缺少线程安全的上下文访问

## 2. 目标（想做成什么样子）

### 核心目标
- **统一上下文**：提供 Context 类管理执行上下文
- **执行栈追踪**：追踪当前执行位置和调用栈
- **参数传递**：传递运行时配置和参数
- **线程安全**：支持多线程安全访问

### 预期效果
- 用户可以创建和传递 Context
- 支持上下文作用域管理
- 支持嵌套上下文
- 支持上下文序列化

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要新增的文件
- `framework/include/nndeploy/dag/context.h` - 上下文管理器
- `framework/source/nndeploy/dag/context.cc` - 上下文实现

### 需要修改的文件
- `framework/include/nndeploy/dag/executor.h` - 集成上下文传递
- `framework/include/nndeploy/dag/graph_runner.h` - 设置和传递上下文
- 各 Executor 子类 - 适配上下文传递

### 不能修改的文件
- 现有的执行逻辑保持不变
- 图结构定义不受影响

### 影响范围
- 所有 Executor 需要适配上下文传递
- Node 执行需要支持上下文访问

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：无统一上下文管理，参数通过构造函数或成员变量传递
- **新方案**：提供 Context 类，统一管理执行上下文
- **核心变化**：新增 Context 系统，Executor 集成上下文传递

### 架构/接口设计

#### 类型定义
```cpp
// 上下文作用域
enum ContextScope {
    kScopeGlobal,      // 全局作用域
    kScopeGraph,       // 图作用域
    kScopeNode,        // 节点作用域
    kScopeLocal        // 本地作用域
};

// 执行栈帧
struct ContextFrame {
    std::string node_name;        // 节点名称
    std::string graph_id;         // 图 ID
    int64_t start_time;           // 开始时间
    ContextScope scope;           // 作用域
    std::map<std::string, std::string> attributes;  // 属性
};

// 上下文配置
struct ContextConfig {
    bool enable_tracing;          // 启用追踪
    bool enable_profiling;        // 启用性能分析
    int max_stack_depth;          // 最大栈深度
    bool auto_propagate;          // 自动传播
};

// 上下文类
class Context {
public:
    Context(const std::string& context_id = "");
    Context(const Context& other);
    ~Context();

    // 创建子上下文（继承父上下文）
    std::shared_ptr<Context> createChild(const std::string& name = "");

    // 获取上下文 ID
    std::string getId() const { return context_id_; }

    // 获取父上下文
    std::shared_ptr<Context> getParent() const { return parent_; }

    // 设置值
    void set(const std::string& key, const std::string& value);
    void set(const std::string& key, int value);
    void set(const std::string& key, double value);
    void set(const std::string& key, const nlohmann::json& value);

    // 获取值
    bool get(const std::string& key, std::string& value) const;
    bool get(const std::string& key, int& value) const;
    bool get(const std::string& key, double& value) const;
    bool get(const std::string& key, nlohmann::json& value) const;

    // 获取或设置默认
    template<typename T>
    T getOrDefault(const std::string& key, const T& default_value) const;

    // 检查键是否存在
    bool has(const std::string& key) const;

    // 删除键
    void remove(const std::string& key);

    // 获取所有键
    std::vector<std::string> getKeys() const;

    // 执行栈管理
    void pushFrame(const ContextFrame& frame);
    void popFrame();
    const std::vector<ContextFrame>& getStack() const { return stack_; }
    ContextFrame* getCurrentFrame();
    ContextFrame* getFrame(size_t depth);

    // 元数据
    void setMetadata(const std::string& key, const std::string& value);
    std::string getMetadata(const std::string& key) const;

    // 序列化
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& json);

    // 清空
    void clear();

    // 合并其他上下文
    void merge(const Context& other);

private:
    std::string context_id_;
    std::shared_ptr<Context> parent_;
    std::map<std::string, nlohmann::json> values_;
    std::vector<ContextFrame> stack_;
    std::map<std::string, std::string> metadata_;
    mutable std::mutex mutex_;
};

// 上下文管理器（单例或全局）
class ContextManager {
public:
    static ContextManager& getInstance();

    // 设置当前上下文（线程局部）
    void setCurrentContext(std::shared_ptr<Context> context);
    std::shared_ptr<Context> getCurrentContext();

    // 创建新上下文并设为当前
    std::shared_ptr<Context> createContext(const std::string& context_id = "");

    // 作用域助手
    std::shared_ptr<ContextScopeGuard> enterScope(const std::string& name);

    // 配置
    void setConfig(const ContextConfig& config);
    const ContextConfig& getConfig() const;

private:
    ContextManager();
    ~ContextManager();

    thread_local static std::shared_ptr<Context> current_context_;
    ContextConfig config_;
    std::mutex mutex_;
};

// 作用域守卫（RAII）
class ContextScopeGuard {
public:
    ContextScopeGuard(std::shared_ptr<Context> context, const std::string& name);
    ~ContextScopeGuard();

    Context* getContext() const { return context_.get(); }

private:
    std::shared_ptr<Context> context_;
    ContextFrame frame_;
};
```

#### API 设计
```cpp
// GraphRunner 集成
class GraphRunner {
public:
    // 设置全局上下文
    void setGlobalContext(std::shared_ptr<Context> context);

    // 获取全局上下文
    std::shared_ptr<Context> getGlobalContext();

    // 创建执行上下文
    std::shared_ptr<Context> createExecutionContext();

protected:
    std::shared_ptr<Context> global_context_;
    std::shared_ptr<Context> execution_context_;
};

// Executor 集成
class Executor {
public:
    // 设置上下文
    void setContext(std::shared_ptr<Context> context);

    // 获取上下文
    std::shared_ptr<Context> getContext() const;

protected:
    // 进入节点作用域
    void enterNodeScope(Node* node);

    // 退出节点作用域
    void exitNodeScope(Node* node);

    std::shared_ptr<Context> context_;
};

// Node 扩展
class Node {
public:
    // 获取当前上下文
    std::shared_ptr<Context> getCurrentContext() const;

    // 从上下文获取值
    template<typename T>
    T getContextValue(const std::string& key, const T& default_value) const;

    // 向上下文设置值
    void setContextValue(const std::string& key, const std::string& value);
    void setContextValue(const std::string& key, int value);
    void setContextValue(const std::string& key, double value);

    // 在执行时自动作用域
    virtual Status runWithContext(std::shared_ptr<Context> context);
};

// 便捷宏
#define NNDEPLOY_CONTEXT_SCOPE(name) \
    auto _nndeploy_scope_guard = \
        nndeploy::dag::ContextManager::getInstance().enterScope(name);

#define NNDEPLOY_SET_CONTEXT(key, value) \
    nndeploy::dag::ContextManager::getInstance().getCurrentContext()->set(key, value);

#define NNDEPLOY_GET_CONTEXT(key, value) \
    nndeploy::dag::ContextManager::getInstance().getCurrentContext()->get(key, value);
```

### 核心操作流程

#### 上下文创建和传递
```
┌─────────────────┐
│  GraphRunner    │
│  创建上下文      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Context        │
│  set() 参数     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Executor       │
│  setContext()   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Node           │
│  访问上下文      │
└─────────────────┘
```

#### 作用域管理
```
┌─────────────────┐
│  enterScope()   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  pushFrame()   │
│  压栈           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  执行逻辑       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ~ScopeGuard() │
│  popFrame()    │
│  出栈           │
└─────────────────┘
```

### 技术细节
- 使用 thread_local 存储线程局部上下文
- 上下文 ID 使用 UUID 生成
- 支持上下文继承和覆盖
- 使用互斥锁保护并发访问
- 上下文清理策略：
  - 作用域守卫自动清理（RAII）
  - 线程退出时自动清理 thread_local
  - 提供手动清理 API
- 上下文值传递使用智能指针管理生命周期

## 5. 实施步骤

### Step 1: 定义 Context 类
- 定义 ContextScope、ContextFrame 结构
- 实现 Context 类
- 涉及文件：`framework/include/nndeploy/dag/context.h`, `framework/source/nndeploy/dag/context.cc`

### Step 2: 实现上下文管理器
- 实现 ContextManager 单例
- 实现线程局部上下文存储
- 实现作用域守卫
- 涉及文件：`framework/include/nndeploy/dag/context.h`, `framework/source/nndeploy/dag/context.cc`

### Step 3: 集成 GraphRunner
- 实现 setGlobalContext()
- 实现执行上下文创建
- 涉及文件：`framework/include/nndeploy/dag/graph_runner.h`, `framework/source/nndeploy/dag/graph_runner.cc`

### Step 4: 集成 Executor
- 实现上下文设置和获取
- 实现作用域管理
- 涉及文件：`framework/include/nndeploy/dag/executor.h`, `framework/source/nndeploy/dag/executor.cc`

### Step 5: 扩展 Node
- 实现上下文访问接口
- 实现作用域自动管理
- 涉及文件：`framework/include/nndeploy/dag/node.h`, `framework/source/nndeploy/dag/node.cc`

### Step 6: 测试和文档
- 编写单元测试
- 编写使用示例
- 编写便捷宏示例
- 涉及文件：`test/`, `docs/`

### 兼容性与迁移
- 上下文功能可选使用
- 不设置上下文时行为不变
- 现有代码无需修改

## 6. 验收标准

### 功能测试
- **测试用例 1**：创建和传递上下文正确
- **测试用例 2**：子上下文继承父上下文正确
- **测试用例 3**：上下文值设置和获取正确
- **测试用例 4**：作用域守卫正确管理栈
- **测试用例 5**：多线程上下文隔离正确
- **测试用例 6**：上下文序列化/反序列化正确
- **测试用例 7**：嵌套作用域正确
- **测试用例 8**：上下文合并正确

### 代码质量
- 通过 GoogleTest 单元测试
- 代码覆盖率 > 80%
- 符合项目代码规范
- 头文件注释完善

### 回归测试
- 现有 DAG 执行功能正常运行
- 不使用上下文时无性能影响
- 各种 Executor 仍能正常工作

### 性能与可维护性
- 上下文访问延迟 < 0.1ms
- 作用域管理开销 < 1ms
- 代码结构清晰，易于扩展

### 文档与示例
- API 文档完整
- 提供使用示例
- 提供便捷宏文档

## 7. 其他说明

### 相关资源
- Go context 包设计
- OpenTelemetry Context 概念

### 风险与应对
- **风险**：上下文传递导致耦合
  - **应对**：文档明确使用场景，推荐最小依赖，提供上下文隔离机制
- **风险**：线程局部上下文内存泄漏
  - **应对**：使用智能指针，作用域守卫自动清理，定期检测泄漏
- **风险**：作用域嵌套过深
  - **应对**：设置最大深度限制（默认 100），检测异常嵌套
- **风险**：上下文竞争条件
  - **应对**：使用读写锁保护共享上下文，提供只读视图

### 依赖关系
- 依赖：无（独立功能）
- 被依赖：调试工具、回调系统

### 扩展方向
- 支持上下文传播到子进程
- 支持分布式上下文传递
- 支持上下文采样和追踪
- 支持上下文性能分析
