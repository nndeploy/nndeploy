---
name: feature_streaming
title: 流式输出（Streaming）机制
description: 支持DAG执行过程中实时推送中间结果，提供Token级别、消息级别、事件级别流式输出，支持WebSocket和SSE传输
category: [feature]
difficulty: medium
priority: P2
status: planned
version: 1.0.0
tags: [dag, streaming, llm, websocket, sse, real-time]
estimated_time: 8h
files_affected: [framework/include/nndeploy/dag/stream.h, framework/source/nndeploy/dag/stream.cc, framework/include/nndeploy/dag/node.h, framework/include/nndeploy/dag/executor.h]
---

# Feature: 流式输出（Streaming）机制

## 1. 背景（是什么 && 为什么）

### 现状分析
- **当前状态**：nndeploy DAG 执行完成后一次性返回结果，无法实时推送中间结果
- **存在问题**：前端无法展示执行进度，LLM 生成等长耗时任务用户体验差
- **需求痛点**：实时进度展示、Token 级别的流式输出、WebSocket 前端集成

### 设计问题
- 没有流式数据推送接口
- 没有统一的流式数据格式
- Node 执行过程中无法推送中间结果
- 缺少事件流（Event Stream）支持

## 2. 目标（想做成什么样子）

### 核心目标
- **流式推送**：支持 Node 执行过程中推送中间结果
- **多种格式**：支持 Token 级别、消息级别、事件级别流式输出
- **多种传输方式**：支持回调、WebSocket、EventStream
- **前端集成**：提供前端友好的 API

### 预期效果
- Node 可以调用 Stream API 推送进度
- 支持增量数据更新
- 支持流式结束标志
- 内置常用流式处理器

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要新增的文件
- `framework/include/nndeploy/dag/stream.h` - 流式处理器接口和类型定义
- `framework/source/nndeploy/dag/stream.cc` - 流式系统实现

### 需要修改的文件
- `framework/include/nndeploy/dag/node.h` - 添加流式推送接口
- `framework/include/nndeploy/dag/executor.h` - 集成流式触发
- `framework/include/nndeploy/dag/graph_runner.h` - 注册流式处理器

### 不能修改的文件
- 现有的 Edge 数据传输机制保持不变
- Device、Tensor 等底层模块不做修改

### 影响范围
- 流式推送可能增加网络开销
- 前端需要适配流式数据格式

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：执行完成后一次性返回结果
- **新方案**：执行过程中实时推送中间结果
- **核心变化**：新增 StreamHandler 接口，Node 支持流式推送

### 架构/接口设计

#### 类型定义
```cpp
// 流式数据类型
enum StreamDataType {
    kStreamTypeToken,       // Token 级别（LLM）
    kStreamTypeMessage,     // 消息级别
    kStreamTypeEvent,       // 事件级别
    kStreamTypeProgress,    // 进度信息
    kStreamTypeData,        // 通用数据
    kStreamTypeEnd,         // 流结束标志
};

// 流式数据块
struct StreamChunk {
    std::string stream_id;          // 流 ID
    StreamDataType type;            // 数据类型
    std::string node_name;          // 节点名称
    int64_t timestamp;              // 时间戳
    int sequence;                   // 序列号

    // 数据内容
    std::string content;            // 文本内容
    nlohmann::json data;            // JSON 数据
    float progress;                 // 进度（0.0-1.0）

    // 元数据
    nlohmann::json metadata;
    bool is_last;                   // 是否最后一个块
};

// 流式处理器基类
class StreamHandler {
public:
    virtual ~StreamHandler() = default;

    // 处理流式数据
    virtual void onChunk(const StreamChunk& chunk) = 0;

    // 流开始
    virtual void onStreamStart(const std::string& stream_id) {}

    // 流结束
    virtual void onStreamEnd(const std::string& stream_id) {}

    // 获取处理器名称
    virtual std::string getName() const { return "StreamHandler"; }

    // 是否异步处理
    virtual bool isAsync() const { return false; }
};
```

#### 内置流式处理器
```cpp
// 回调流式处理器
class CallbackStreamHandler : public StreamHandler {
public:
    CallbackStreamHandler(std::function<void(const StreamChunk&)> callback);

    virtual void onChunk(const StreamChunk& chunk) override;

private:
    std::function<void(const StreamChunk&)> callback_;
};

// 日志流式处理器
class LoggingStreamHandler : public StreamHandler {
public:
    LoggingStreamHandler(const std::string& log_file = "");
    virtual ~LoggingStreamHandler();

    virtual void onChunk(const StreamChunk& chunk) override;

private:
    std::ofstream log_file_;
};

// WebSocket 流式处理器
class WebSocketStreamHandler : public StreamHandler {
public:
    WebSocketStreamHandler(const std::string& url);
    virtual ~WebSocketStreamHandler();

    virtual void onChunk(const StreamChunk& chunk) override;
    virtual void onStreamStart(const std::string& stream_id) override;
    virtual void onStreamEnd(const std::string& stream_id) override;

    // 异步处理
    virtual bool isAsync() const override { return true; }

    bool connect();
    void disconnect();

private:
    void sendChunk(const StreamChunk& chunk);
    std::string url_;
    std::shared_ptr<WebSocketClient> client_;
};

// EventStream 处理器（SSE）
// 注意：建议使用线程安全的输出流，或在单线程环境使用
class EventStreamHandler : public StreamHandler {
public:
    // 输出流包装器（线程安全版本）
    class ThreadSafeOutput {
    public:
        explicit ThreadSafeOutput(std::ostream& output);
        void write(const std::string& data);
        void flush();

    private:
        std::ostream& output_;
        std::mutex mutex_;
    };

    EventStreamHandler(std::ostream& output);
    EventStreamHandler(std::shared_ptr<ThreadSafeOutput> output);

    virtual void onChunk(const StreamChunk& chunk) override;
    virtual void onStreamEnd(const std::string& stream_id) override;

private:
    void writeEvent(const std::string& event, const std::string& data);
    std::shared_ptr<ThreadSafeOutput> output_;
    std::mutex mutex_;
};
```

#### API 设计
```cpp
// Node 扩展
class Node {
public:
    // 流式推送 Token
    bool streamToken(const std::string& token);

    // 流式推送消息
    bool streamMessage(const std::string& message);

    // 流式推送进度
    bool streamProgress(float progress, const std::string& message = "");

    // 流式推送事件
    bool streamEvent(const std::string& event, const nlohmann::json& data);

    // 流式推送通用数据
    bool streamData(const nlohmann::json& data);

    // 结束流
    bool endStream();

protected:
    // 获取流 ID
    std::string getStreamId() const;

    // 生成流式数据块
    StreamChunk createChunk(StreamDataType type, const std::string& content);
};

// 流管理器
class StreamManager {
public:
    // 注册处理器
    void registerHandler(std::shared_ptr<StreamHandler> handler);

    // 注销处理器
    void unregisterHandler(const std::string& name);

    // 推送流式数据
    void push(const StreamChunk& chunk);

    // 创建新流
    std::string createStream(const std::string& node_name);

    // 结束流
    void endStream(const std::string& stream_id);

    // 获取处理器列表
    std::vector<std::shared_ptr<StreamHandler>> getHandlers() const;

private:
    void triggerStreamStart(const std::string& stream_id);
    void triggerStreamEnd(const std::string& stream_id);

    std::map<std::string, std::shared_ptr<StreamHandler>> handlers_;
    std::map<std::string, int> stream_sequences_;
    std::mutex mutex_;
};

// GraphRunner 集成
class GraphRunner {
public:
    // 注册流式处理器
    void registerStreamHandler(std::shared_ptr<StreamHandler> handler);

    // 获取流管理器
    StreamManager* getStreamManager();

    // 启用流式输出
    void enableStreaming(bool enable = true);

protected:
    StreamManager stream_manager_;
    bool streaming_enabled_;
};
```

### 核心操作流程

#### 流式推送流程
```
┌─────────────────┐
│  Node 执行      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  streamToken()  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  createChunk()  │
│  生成流式数据块 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  StreamManager  │
│  push()         │
└────────┬────────┘
         │
         ├──▶ Handler 1.onChunk()
         ├──▶ Handler 2.onChunk()
         └──▶ Handler 3.onChunk()
```

#### WebSocket 集成流程
```
┌─────────────────┐       ┌─────────────────┐
│  nndeploy DAG   │       │  前端           │
└────────┬────────┘       └────────┬────────┘
         │                        │
         │  1. WebSocket Connect   │
         │◀───────────────────────│
         │                        │
         │  2. Stream Start        │
         │◀───────────────────────│
         │                        │
         │  3. Stream Chunks       │
         │◀───────────────────────│
         │                        │
         │  4. Stream End          │
         │◀───────────────────────│
```

### 技术细节
- 流 ID 使用 UUID 生成
- WebSocket 库选择：优先使用 uWebSockets（高性能），备选 WebSocket++
- EventStream 遵循 SSE（Server-Sent Events）标准
- 支持流式数据压缩（gzip/deflate）
- 流式数据顺序保证：使用序列号，前端乱序重排
- 多租户隔离：流 ID 包含 session_id，支持按会话过滤

## 5. 实施步骤

### Step 1: 定义流式系统基础类型
- 定义 StreamDataType 枚举
- 定义 StreamChunk 结构
- 定义 StreamHandler 基类
- 涉及文件：`framework/include/nndeploy/dag/stream.h`

### Step 2: 实现 StreamManager
- 实现处理器注册/注销
- 实现流式数据推送逻辑
- 实现流生命周期管理
- 涉及文件：`framework/include/nndeploy/dag/stream.h`, `framework/source/nndeploy/dag/stream.cc`

### Step 3: 实现内置流式处理器
- 实现 CallbackStreamHandler
- 实现 LoggingStreamHandler
- 实现 WebSocketStreamHandler
- 实现 EventStreamHandler
- 涉及文件：`framework/include/nndeploy/dag/stream.h`, `framework/source/nndeploy/dag/stream.cc`

### Step 4: 扩展 Node 流式接口
- 添加 streamToken() 等方法
- 实现 createChunk() 辅助方法
- 涉及文件：`framework/include/nndeploy/dag/node.h`, `framework/source/nndeploy/dag/node.cc`

### Step 5: 集成 GraphRunner
- 实现流式处理器注册接口
- 实现流式启用控制
- 涉及文件：`framework/include/nndeploy/dag/graph_runner.h`, `framework/source/nndeploy/dag/graph_runner.cc`

### Step 6: Python 绑定和 Demo
- 添加 Python 流式 API
- 创建 WebSocket 服务器 Demo
- 创建前端集成示例
- 涉及文件：`python/`, `demo/`

### 兼容性与迁移
- 流式功能可选启用
- 不启用流式时无性能影响
- 现有代码无需修改

## 6. 验收标准

### 功能测试
- **测试用例 1**：Node 正确推送 Token 级别数据
- **测试用例 2**：多个处理器同时接收流式数据
- **测试用例 3**：WebSocket 处理器正常工作
- **测试用例 4**：EventStream 处理器正确生成 SSE 格式
- **测试用例 5**：流式数据按顺序到达
- **测试用例 6**：流结束标志正确触发
- **测试用例 7**：进度更新正常工作
- **测试用例 8**：前端能正确接收并展示流式数据

### 代码质量
- 通过 GoogleTest 单元测试
- 代码覆盖率 > 80%
- 符合项目代码规范
- 头文件注释完善

### 回归测试
- 现有 DAG 执行功能正常运行
- 不启用流式时执行性能无明显变化
- 各种 Executor 仍能正常工作

### 性能与可维护性
- 流式推送延迟 < 5ms
- 支持高并发流式推送
- 代码结构清晰，易于扩展新的处理器

### 文档与示例
- API 文档完整
- 提供使用示例
- 提供前端集成示例
- 提供 WebSocket 服务器 Demo

## 7. 其他说明

### 相关资源
- Server-Sent Events (SSE) 标准
- WebSocket 协议文档
- LLM 流式输出最佳实践

### 风险与应对
- **风险**：WebSocket 连接不稳定
  - **应对**：实现自动重连机制，心跳检测
- **风险**：流式数据顺序错乱
  - **应对**：使用序列号，前端重新排序
- **风险**：高并发时性能问题
  - **应对**：使用批处理，异步推送

### 依赖关系
- 依赖：回调系统（可选）
- 被依赖：LLM 插件

### 扩展方向
- 支持流式数据压缩
- 支持流式数据过滤
- 支持流式数据聚合
- 支持多租户流式隔离
