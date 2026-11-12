---
name: feature_event_driven
title: 事件驱动执行
description: 实现基于事件的DAG执行模式，支持事件定义、订阅发布机制、事件驱动执行器和异步事件处理
category: [feature]
difficulty: medium
priority: P2
status: planned
version: 1.0.0
tags: [dag, event, executor, event-driven, async, pub-sub]
estimated_time: 8h
files_affected: [framework/include/nndeploy/dag/event.h, framework/source/nndeploy/dag/event.cc, framework/include/nndeploy/dag/event_executor.h, framework/source/nndeploy/dag/event_executor.cc]
---

# Feature: 事件驱动执行

## 1. 背景（是什么 && 为什么）

### 现状分析
- **当前状态**：nndeploy DAG 是命令式执行，由 Executor 驱动图执行
- **存在问题**：不支持事件驱动的执行模式，无法响应外部事件
- **需求痛点**：需要基于事件的执行触发、支持外部事件输入、异步事件处理

### 设计问题
- 没有事件定义和管理
- 没有事件驱动执行器
- 不支持事件订阅/发布机制
- 无法与现有 Executor 兼容

## 2. 目标（想做成什么样子）

### 核心目标
- **事件定义**：提供标准化的 Event 类
- **事件驱动**：实现 EventDrivenExecutor
- **订阅发布**：支持事件订阅/发布机制
- **异步处理**：支持异步事件处理

### 预期效果
- 用户可以定义和处理自定义事件
- 节点可以订阅和响应事件
- 支持外部事件触发图执行
- 与现有 Executor 兼容

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要新增的文件
- `framework/include/nndeploy/dag/event.h` - 事件定义和事件总线
- `framework/source/nndeploy/dag/event.cc` - 事件实现
- `framework/include/nndeploy/dag/event_executor.h` - 事件驱动执行器
- `framework/source/nndeploy/dag/event_executor.cc` - 事件执行器实现

### 需要修改的文件
- `framework/include/nndeploy/dag/node.h` - 添加事件处理接口（可选）

### 不能修改的文件
- 现有的 Executor 类保持不变
- 现有的执行逻辑不受影响

### 影响范围
- 主要影响需要事件驱动的场景
- 不影响现有命令式执行

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：只能命令式执行图
- **新方案**：支持事件驱动的执行模式
- **核心变化**：新增 Event 系统和 EventDrivenExecutor

### 架构/接口设计

#### 类型定义
```cpp
// 事件类型
enum EventType {
    kEventTypeCustom,       // 自定义事件
    kEventTypeData,         // 数据事件
    kEventTypeControl,      // 控制事件
    kEventTypeTimer,        // 定时器事件
    kEventTypeSignal,       // 信号事件
    kEventTypeUser          // 用户事件
};

// 事件优先级
enum EventPriority {
    kPriorityLow = 0,
    kPriorityNormal = 50,
    kPriorityHigh = 100,
    kPriorityCritical = 200
};

// 事件类
class Event {
public:
    Event(const std::string& name,
          EventType type = kEventTypeCustom,
          EventPriority priority = kPriorityNormal);

    // 访问器
    std::string getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }

    EventType getType() const { return type_; }
    void setType(EventType type) { type_ = type; }

    EventPriority getPriority() const { return priority_; }
    void setPriority(EventPriority priority) { priority_ = priority; }

    int64_t getTimestamp() const { return timestamp_; }

    // 数据访问
    void setData(const std::string& key, const nlohmann::json& value);
    bool getData(const std::string& key, nlohmann::json& value) const;
    const nlohmann::json& getAllData() const { return data_; }

    // 序列化
    nlohmann::json toJson() const;
    static Event fromJson(const nlohmann::json& json);

    // 比较（用于优先级队列）
    bool operator<(const Event& other) const;

private:
    std::string name_;
    EventType type_;
    EventPriority priority_;
    int64_t timestamp_;
    nlohmann::json data_;
};

// 事件处理器基类
class EventHandler {
public:
    virtual ~EventHandler() = default;

    // 处理事件
    virtual void handle(const Event& event) = 0;

    // 是否可以处理指定事件
    virtual bool canHandle(const Event& event) const;

    // 获取处理器名称
    virtual std::string getName() const = 0;

    // 是否异步处理
    virtual bool isAsync() const { return false; }
};

// 事件总线
class EventBus {
public:
    static EventBus& getInstance();

    // 订阅事件
    void subscribe(const std::string& event_name,
                   std::shared_ptr<EventHandler> handler);

    // 取消订阅
    void unsubscribe(const std::string& event_name,
                     const std::shared_ptr<EventHandler>& handler);

    // 发布事件
    void publish(const Event& event);

    // 同步发布（等待所有处理器完成）
    void publishSync(const Event& event);

    // 异步发布
    void publishAsync(const Event& event);

    // 设置线程池大小
    void setThreadPoolSize(size_t size);

private:
    EventBus();
    ~EventBus();

    std::map<std::string, std::vector<std::shared_ptr<EventHandler>>> handlers_;
    std::shared_ptr<ThreadPool> thread_pool_;
    std::mutex mutex_;
};

// 事件节点（可以处理事件的 Node）
class EventNode : public Node {
public:
    EventNode(const std::string& name = "event_node");

    // 订阅事件
    void subscribe(const std::string& event_name);

    // 取消订阅
    void unsubscribe(const std::string& event_name);

    // 处理事件（由子类实现）
    virtual void onEvent(const Event& event) {}

    // 触发事件
    void emit(const Event& event);

    // 获取待处理事件队列
    std::queue<Event> getPendingEvents();
    void clearPendingEvents();

protected:
    std::set<std::string> subscribed_events_;
    std::queue<Event> pending_events_;
    std::mutex events_mutex_;
};
```

#### 事件驱动执行器
```cpp
// 事件驱动执行器
class EventDrivenExecutor : public Executor {
public:
    EventDrivenExecutor();

    // 启动执行器
    void start();

    // 停止执行器
    void stop();

    // 是否运行中
    bool isRunning() const { return running_; }

    // 提交事件
    void submitEvent(const Event& event);

    // 注册事件节点
    void registerEventNode(EventNode* node);

    // 注销事件节点
    void unregisterEventNode(const std::string& node_name);

    // 设置最大并发数
    void setMaxConcurrency(size_t max_concurrency);

protected:
    // 主循环
    void eventLoop();

    // 处理事件
    void processEvent(const Event& event);

    // 查找订阅的节点
    std::vector<EventNode*> findSubscribers(const std::string& event_name);

private:
    std::thread event_thread_;
    std::priority_queue<Event> event_queue_;
    std::map<std::string, std::vector<EventNode*>> subscriptions_;
    std::map<std::string, EventNode*> event_nodes_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_;
    size_t max_concurrency_;
    std::shared_ptr<ThreadPool> worker_pool_;
};
```

#### API 设计
```cpp
// 便捷事件创建
namespace nndeploy {
namespace dag {

// 创建数据事件
Event createDataEvent(const std::string& name,
                       const nlohmann::json& data,
                       EventPriority priority = kPriorityNormal);

// 创建控制事件
Event createControlEvent(const std::string& name,
                         const std::string& action);

// 创建定时器事件
Event createTimerEvent(const std::string& name, int64_t delay_ms);

// 发布事件
inline void publishEvent(const Event& event) {
    EventBus::getInstance().publish(event);
}

}} // namespace nndeploy::dag

// 便捷宏
#define NNDEPLOY_SUBSCRIBE_EVENT(name) \
    this->subscribe(name);

#define NNDEPLOY_EMIT_EVENT(name) \
    this->emit(nndeploy::dag::createDataEvent(name, {}));

#define NNDEPLOY_ON_EVENT(event_var) \
    virtual void onEvent(const Event& event_var) override
```

### 核心操作流程

#### 事件发布和处理流程
```
┌─────────────────┐
│  EventNode      │
│  emit()         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EventBus      │
│  publish()     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  查找订阅者      │
└────────┬────────┘
         │
         ├──▶ Handler 1.handle()
         ├──▶ Handler 2.handle()
         └──▶ Handler 3.handle()
```

#### 事件驱动执行流程
```
┌─────────────────┐
│  EventDriven    │
│  Executor       │
│  start()        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  eventLoop()   │
│  等待事件       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  processEvent() │
│  分发给订阅者    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EventNode      │
│  onEvent()      │
└─────────────────┘
```

### 技术细节
- 使用优先级队列管理事件（基于 EventPriority）
- 事件 ID 使用 UUID
- 支持事件过滤（名称、类型、优先级）
- 支持事件超时处理（设置 event.timeout_ms）
- 事件队列管理：
  - 最大队列长度限制（防止内存溢出）
  - 满队列策略（丢弃/阻塞/降级）
  - 事件优先级调度
- 内存泄漏防护：
  - 使用智能指针管理事件处理器
  - 定期清理无效事件
  - 提供事件队列状态监控

## 5. 实施步骤

### Step 1: 定义 Event 类
- 定义 EventType、EventPriority 枚举
- 实现 Event 类
- 涉及文件：`framework/include/nndeploy/dag/event.h`, `framework/source/nndeploy/dag/event.cc`

### Step 2: 实现 EventBus
- 实现订阅/取消订阅
- 实现事件发布逻辑
- 涉及文件：`framework/include/nndeploy/dag/event.h`, `framework/source/nndeploy/dag/event.cc`

### Step 3: 实现 EventNode
- 扩展 Node 类
- 实现事件订阅和处理
- 涉及文件：`framework/include/nndeploy/dag/event.h`, `framework/source/nndeploy/dag/event.cc`

### Step 4: 实现 EventDrivenExecutor
- 实现事件驱动执行逻辑
- 实现事件分发
- 涉及文件：`framework/include/nndeploy/dag/event_executor.h`, `framework/source/nndeploy/dag/event_executor.cc`

### Step 5: 测试和文档
- 编写单元测试
- 编写使用示例
- 编写事件驱动 Demo
- 涉及文件：`test/`, `docs/`, `demo/`

### 兼容性与迁移
- 事件驱动功能可选使用
- 不影响现有 Executor
- 现有代码无需修改

## 6. 验收标准

### 功能测试
- **测试用例 1**：创建和发布事件正确
- **测试用例 2**：事件订阅和取消订阅正确
- **测试用例 3**：事件处理器正确接收事件
- **测试用例 4**：事件优先级排序正确
- **测试用例 5**：EventDrivenExecutor 正确处理事件
- **测试用例 6**：EventNode 正确订阅和响应事件
- **测试用例 7**：异步事件处理不阻塞
- **测试用例 8**：事件队列无泄漏

### 代码质量
- 通过 GoogleTest 单元测试
- 代码覆盖率 > 80%
- 符合项目代码规范
- 头文件注释完善

### 回归测试
- 现有 DAG 执行功能正常运行
- 现有 Executor 仍能正常工作
- 不使用事件驱动时无影响

### 性能与可维护性
- 事件发布延迟 < 1ms
- 事件处理延迟 < 10ms
- 代码结构清晰，易于扩展

### 文档与示例
- API 文档完整
- 提供使用示例
- 提供事件驱动 Demo

## 7. 其他说明

### 相关资源
- 观察者模式
- 事件驱动架构模式

### 风险与应对
- **风险**：事件队列阻塞
  - **应对**：使用优先级队列，设置超时，满队列策略
- **风险**：事件处理异常导致系统崩溃
  - **应对**：捕获所有异常，记录日志，异常隔离（单事件失败不影响其他）
- **风险**：内存泄漏（事件未处理）
  - **应对**：使用智能指针，定期清理，队列监控
- **风险**：事件风暴（大量事件涌入）
  - **应对**：限流机制，事件合并，优先级丢弃

### 依赖关系
- 依赖：ThreadPool
- 被依赖：动态图修改

### 扩展方向
- 支持事件持久化
- 支持事件重放
- 支持分布式事件总线
- 支持事件规则引擎
