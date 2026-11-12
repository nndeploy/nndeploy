---
name: feature_callback_system
title: 完整回调系统
description: 为DAG执行节点提供生命周期钩子、多处理器支持和异步回调能力，用于监控、日志、追踪等场景
category: [feature]
difficulty: medium
priority: P2
status: planned
version: 1.0.0
tags: [dag, callback, executor, lifecycle, monitoring, tracing]
estimated_time: 8h
files_affected: [framework/include/nndeploy/dag/callback.h, framework/source/nndeploy/dag/callback.cc, framework/include/nndeploy/dag/executor.h, framework/include/nndeploy/dag/graph_runner.h]
---

# Feature: 完整回调系统

## 1. 背景（是什么 && 为什么）

### 现状分析
- **当前状态**：nndeploy DAG 模块定义了回调函数类型，但没有完整的回调管理系统
- **存在问题**：无法在节点执行生命周期中插入自定义逻辑，难以实现监控、日志、追踪等功能
- **需求痛点**：实际应用中需要监听节点执行前后的状态、捕获错误、收集指标等

### 设计问题
- 没有统一的回调处理器基类
- 没有生命周期钩子定义
- Executor 层面缺少回调触发逻辑
- 不支持多个回调处理器注册

## 2. 目标（想做成什么样子）

### 核心目标
- **生命周期钩子**：在节点执行前、执行后、出错时触发回调
- **多处理器支持**：支持注册多个回调处理器
- **异步回调**：支持同步和异步两种回调执行模式
- **链式处理**：支持回调链管理

### 预期效果
- 用户可以继承 CallbackHandler 实现自定义逻辑
- 支持事件类型过滤
- 支持回调优先级
- 内置常用回调处理器（日志、监控、追踪）

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要新增的文件
- `framework/include/nndeploy/dag/callback.h` - 回调处理器基类和事件定义
- `framework/source/nndeploy/dag/callback.cc` - 回调系统实现

### 需要修改的文件
- `framework/include/nndeploy/dag/executor.h` - 集成回调触发逻辑
- `framework/source/nndeploy/dag/executor.cc` - 实现回调触发
- `framework/include/nndeploy/dag/graph_runner.h` - 注册回调处理器

### 不能修改的文件
- 现有的 Node 执行逻辑保持不变
- 现有的 Edge 数据传输不受影响

### 影响范围
- 所有 Executor 子类需要支持回调触发
- 回调执行可能影响执行性能

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：只有简单的回调函数类型定义，没有管理系统
- **新方案**：完整的回调系统，支持生命周期钩子和多处理器
- **核心变化**：新增 CallbackHandler 基类，Executor 集成回调触发逻辑

### 架构/接口设计

#### 类型定义
```cpp
// 事件类型
enum CallbackEventType {
    kEventNodeStart,       // 节点开始执行
    kEventNodeEnd,         // 节点执行结束
    kEventNodeError,       // 节点执行出错
    kEventGraphStart,      // 图开始执行
    kEventGraphEnd,        // 图执行结束
    kEventCheckpointSaved, // 检查点保存
    kEventHumanFeedback     // 人类反馈事件
};

// 回调事件数据
struct CallbackEvent {
    CallbackEventType type;       // 事件类型
    std::string graph_id;         // 图 ID
    std::string node_name;        // 节点名称
    int64_t timestamp;            // 时间戳

    // 事件特有数据
    nlohmann::json data;          // 通用数据

    // 错误信息（仅用于错误事件）
    Status error_status;

    // 元数据
    nlohmann::json metadata;
};

// 回调处理器基类
class CallbackHandler {
public:
    virtual ~CallbackHandler() = default;

    // 节点开始
    virtual void onNodeStart(const CallbackEvent& event) {}

    // 节点结束
    virtual void onNodeEnd(const CallbackEvent& event) {}

    // 节点错误
    virtual void onNodeError(const CallbackEvent& event) {}

    // 图开始
    virtual void onGraphStart(const CallbackEvent& event) {}

    // 图结束
    virtual void onGraphEnd(const CallbackEvent& event) {}

    // 检查点保存
    virtual void onCheckpointSaved(const CallbackEvent& event) {}

    // 人类反馈
    virtual void onHumanFeedback(const CallbackEvent& event) {}

    // 获取处理器名称
    virtual std::string getName() const { return "CallbackHandler"; }

    // 是否异步执行
    virtual bool isAsync() const { return false; }

    // 优先级（数值越小优先级越高）
    virtual int getPriority() const { return 100; }
};
```

#### 内置回调处理器
```cpp
// 日志回调处理器
class LoggingCallback : public CallbackHandler {
public:
    LoggingCallback(const std::string& log_file = "");
    virtual ~LoggingCallback();

    virtual void onNodeStart(const CallbackEvent& event) override;
    virtual void onNodeEnd(const CallbackEvent& event) override;
    virtual void onNodeError(const CallbackEvent& event) override;

    virtual std::string getName() const override { return "LoggingCallback"; }

private:
    void writeLog(const std::string& message);
    std::ofstream log_file_;
};

// 监控回调处理器
class MetricsCallback : public CallbackHandler {
public:
    virtual void onNodeStart(const CallbackEvent& event) override;
    virtual void onNodeEnd(const CallbackEvent& event) override;

    virtual std::string getName() const override { return "MetricsCallback"; }

    // 获取指标
    nlohmann::json getMetrics() const;
    void resetMetrics();

private:
    struct NodeMetrics {
        int call_count;
        int64_t total_time_us;
        int64_t max_time_us;
        int64_t min_time_us;
    };
    std::map<std::string, NodeMetrics> metrics_;
    std::mutex mutex_;
};

// 追踪回调处理器（用于 LangSmith 集成）
class TracingCallback : public CallbackHandler {
public:
    TracingCallback(const std::string& api_key = "");
    virtual ~TracingCallback();

    virtual void onNodeStart(const CallbackEvent& event) override;
    virtual void onNodeEnd(const CallbackEvent& event) override;
    virtual void onNodeError(const CallbackEvent& event) override;

    virtual std::string getName() const override { return "TracingCallback"; }

    // 是否异步执行
    virtual bool isAsync() const override { return true; }

private:
    void sendTrace(const CallbackEvent& event);
    std::string api_key_;
    std::string trace_id_;
};
```

#### API 设计
```cpp
// 回调管理器
class CallbackManager {
public:
    // 注册回调处理器
    void registerHandler(std::shared_ptr<CallbackHandler> handler);

    // 注销回调处理器
    void unregisterHandler(const std::string& name);

    // 清空所有处理器
    void clear();

    // 触发回调
    void trigger(const CallbackEvent& event);

    // 获取处理器列表
    std::vector<std::shared_ptr<CallbackHandler>> getHandlers() const;

private:
    void sortHandlers();

    std::vector<std::shared_ptr<CallbackHandler>> handlers_;
    std::mutex mutex_;
    bool sorted_;
};

// Executor 集成
class Executor {
public:
    // 设置回调管理器
    void setCallbackManager(CallbackManager* manager);

protected:
    // 触发节点开始回调
    void triggerNodeStart(Node* node);

    // 触发节点结束回调
    void triggerNodeEnd(Node* node, const Status& status);

    // 触发节点错误回调
    void triggerNodeError(Node* node, const Status& status);

    CallbackManager* callback_manager_;
};

// GraphRunner 集成
class GraphRunner {
public:
    // 注册回调处理器
    void registerCallback(std::shared_ptr<CallbackHandler> handler);

    // 获取回调管理器
    CallbackManager* getCallbackManager();

protected:
    // 触发图开始回调
    void triggerGraphStart();

    // 触发图结束回调
    void triggerGraphEnd(const Status& status);

    CallbackManager callback_manager_;
};
```

### 核心操作流程

#### 回调触发流程
```
┌─────────────────┐
│  GraphRunner    │
│  开始执行       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  triggerGraph   │
│  Start()        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CallbackManager│
│  trigger()      │
└────────┬────────┘
         │
         ├──▶ Handler 1.onGraphStart()
         ├──▶ Handler 2.onGraphStart()
         └──▶ Handler 3.onGraphStart()

         │
         ▼
┌─────────────────┐
│  执行节点       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  triggerNode   │
│  Start()       │
└────────┬────────┘
         │
         ├──▶ Handler 1.onNodeStart()
         ├──▶ Handler 2.onNodeStart()
         └──▶ Handler 3.onNodeStart()

         │
         ▼
┌─────────────────┐
│  节点执行       │
└────────┬────────┘
         │
         ├── 成功 ──▶ triggerNodeEnd()
         └── 失败 ──▶ triggerNodeError()
```

### 技术细节
- 异步回调使用线程池执行（复用 nndeploy::thread_pool）
- 回调按优先级顺序执行（数值越小优先级越高）
- 回调执行失败不影响主流程（捕获所有异常）
- 回调处理器之间通过共享 Context 通信（可选）
- 回调链中断策略：单个处理器失败不中断链，记录错误日志

## 5. 实施步骤

### Step 1: 定义回调系统基础类型
- 定义 CallbackEventType 枚举
- 定义 CallbackEvent 结构
- 定义 CallbackHandler 基类
- 涉及文件：`framework/include/nndeploy/dag/callback.h`

### Step 2: 实现 CallbackManager
- 实现回调处理器注册/注销
- 实现回调触发逻辑
- 实现处理器排序（按优先级）
- 涉及文件：`framework/include/nndeploy/dag/callback.h`, `framework/source/nndeploy/dag/callback.cc`

### Step 3: 实现内置回调处理器
- 实现 LoggingCallback
- 实现 MetricsCallback
- 实现 TracingCallback（基础框架）
- 涉及文件：`framework/include/nndeploy/dag/callback.h`, `framework/source/nndeploy/dag/callback.cc`

### Step 4: 集成 Executor
- 添加 CallbackManager 成员
- 实现触发回调的辅助方法
- 在节点执行前后触发回调
- 涉及文件：`framework/include/nndeploy/dag/executor.h`, `framework/source/nndeploy/dag/executor.cc`

### Step 5: 集成 GraphRunner
- 实现回调处理器注册接口
- 在图执行开始/结束时触发回调
- 涉及文件：`framework/include/nndeploy/dag/graph_runner.h`, `framework/source/nndeploy/dag/graph_runner.cc`

### Step 6: 测试和文档
- 编写单元测试
- 编写使用示例
- 编写自定义回调处理器示例
- 涉及文件：`test/`, `docs/`

### 兼容性与迁移
- 回调功能可选启用
- 不注册回调处理器时无性能影响
- 现有代码无需修改

## 6. 验收标准

### 功能测试
- **测试用例 1**：注册单个回调处理器，正确触发所有事件
- **测试用例 2**：注册多个回调处理器，按优先级顺序执行
- **测试用例 3**：异步回调处理器不阻塞主流程
- **测试用例 4**：回调处理器抛出异常不影响主流程
- **测试用例 5**：LoggingCallback 正确记录日志
- **测试用例 6**：MetricsCallback 正确收集指标
- **测试用例 7**：注销回调处理器后不再触发
- **测试用例 8**：自定义回调处理器正常工作

### 代码质量
- 通过 GoogleTest 单元测试
- 代码覆盖率 > 80%
- 符合项目代码规范
- 头文件注释完善

### 回归测试
- 现有 DAG 执行功能正常运行
- 不注册回调时执行性能无明显变化
- 各种 Executor 仍能正常工作

### 性能与可维护性
- 回调执行时间 < 1ms（每个处理器）
- 同步回调总开销 < 10ms
- 代码结构清晰，易于扩展新的处理器

### 文档与示例
- API 文档完整
- 提供使用示例
- 提供自定义回调处理器示例

## 7. 其他说明

### 相关资源
- LangSmith 追踪文档
- OpenTelemetry 标准

### 风险与应对
- **风险**：回调执行时间过长影响主流程
  - **应对**：推荐使用异步回调，设置超时
- **风险**：回调中修改状态导致不一致
  - **应对**：回调只读数据，文档明确说明
- **风险**：回调抛出异常导致崩溃
  - **应对**：捕获所有异常，记录日志

### 依赖关系
- 依赖：无（独立功能）
- 被依赖：流式输出、调试工具

### 扩展方向
- 支持回调过滤器（基于节点名称、事件类型等）
- 支持回调组合（AND/OR 逻辑）
- 支持回调条件执行
- 支持远程回调（HTTP/WebSocket）
