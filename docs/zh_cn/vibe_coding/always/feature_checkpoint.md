---
name: feature_checkpoint
title: 状态持久化（Checkpoint）机制
description: 支持图执行状态的保存和恢复，实现断点续执行能力，支持内存和Redis多种存储后端
category: [feature]
difficulty: medium
priority: P2
status: planned
version: 1.0.0
tags: [dag, checkpoint, persistence, graph, redis]
estimated_time: 8h
files_affected: [framework/include/nndeploy/dag/checkpoint.h, framework/source/nndeploy/dag/checkpoint.cc, framework/include/nndeploy/dag/graph.h, framework/include/nndeploy/dag/graph_runner.h]
---

# Feature: 状态持久化（Checkpoint）机制

## 1. 背景（是什么 && 为什么）

### 现状分析
- **当前状态**：nndeploy DAG 模块支持图的序列化/反序列化，但不支持执行状态的持久化
- **存在问题**：图执行过程中如果中断，无法保存执行进度，中断后必须从头重新执行
- **需求痛点**：长耗时任务、多轮对话、人机协作场景需要断点续执行能力，避免重复计算

### 设计问题
- 没有检查点保存/恢复机制
- 无法保存节点执行状态和数据
- 不支持多种存储后端
- 无法与人类反馈机制配合

## 2. 目标（想做成什么样子）

### 核心目标
- **状态保存**：自动保存图执行状态到检查点
- **状态恢复**：从检查点恢复执行状态，断点续执行
- **多存储后端**：支持内存、Redis 等多种存储方式
- **版本追踪**：支持状态版本管理

### 预期效果
- 用户可以指定保存检查点的时机（节点执行前后、间隔时间等）
- 支持手动触发和自动触发两种模式
- 检查点包含完整的图状态、节点状态、数据快照
- 支持检查点列表查询和删除

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要新增的文件
- `framework/include/nndeploy/dag/checkpoint.h` - 检查点基类和实现类
- `framework/source/nndeploy/dag/checkpoint.cc` - 检查点实现

### 需要修改的文件
- `framework/include/nndeploy/dag/graph.h` - 添加获取/设置状态方法
- `framework/include/nndeploy/dag/graph_runner.h` - 集成检查点保存/恢复逻辑
- `framework/source/nndeploy/dag/graph_runner.cc` - 实现检查点集成
- `framework/include/nndeploy/dag/node.h` - 支持节点状态序列化（如需要）

### 不能修改的文件
- 现有的序列化/反序列化接口保持兼容
- Device、Tensor 等底层模块不做修改

### 影响范围
- 所有 GraphRunner 执行模式需要支持检查点
- 检查点存储依赖序列化机制

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：执行状态只在内存中，进程结束后丢失
- **新方案**：支持状态持久化到存储后端，可恢复执行
- **核心变化**：新增 Checkpoint 机制，GraphRunner 集成检查点逻辑

### 架构/接口设计

#### 类型定义
```cpp
// 检查点配置
enum CheckpointPolicy {
    kCheckpointManual,          // 手动触发
    kCheckpointOnNodeStart,    // 每个节点执行前
    kCheckpointOnNodeEnd,      // 每个节点执行后
    kCheckpointTimeInterval,   // 按时间间隔
    kCheckpointAll             // 所有时机
};

struct CheckpointConfig {
    CheckpointPolicy policy;
    int time_interval_ms;      // 时间间隔（ms）
    size_t max_checkpoints;    // 最大保留检查点数
    bool compress;             // 是否压缩
};

// 检查点数据结构
struct CheckpointData {
    std::string checkpoint_id;        // 检查点 ID
    std::string graph_id;             // 图 ID
    int64_t timestamp;                // 时间戳
    std::string session_id;           // 会话 ID

    // 图状态
    nlohmann::json graph_state;

    // 节点状态（map: node_name -> state）
    std::map<std::string, nlohmann::json> node_states;

    // 边数据（map: edge_name -> data）
    std::map<std::string, nlohmann::json> edge_data;

    // 元数据
    nlohmann::json metadata;
};

// 反馈请求状态（与人类反馈集成）
struct CheckpointFeedbackState {
    bool has_pending_feedback;
    HumanFeedbackRequest feedback_request;
};
```

#### API 设计
```cpp
// 检查点基类
class Checkpoint {
public:
    virtual ~Checkpoint() = default;

    // 保存检查点
    virtual bool save(const CheckpointData& data) = 0;

    // 加载检查点
    virtual bool load(const std::string& checkpoint_id, CheckpointData& data) = 0;

    // 获取检查点列表
    virtual std::vector<CheckpointData> list(const std::string& graph_id) = 0;

    // 删除检查点
    virtual bool remove(const std::string& checkpoint_id) = 0;

    // 清理旧检查点
    virtual bool cleanup(const std::string& graph_id, size_t keep_count) = 0;

protected:
    std::string generateCheckpointId();
    nlohmann::json serializeCheckpoint(const CheckpointData& data);
    bool deserializeCheckpoint(const nlohmann::json& json, CheckpointData& data);
};

// 内存检查点实现
class MemoryCheckpoint : public Checkpoint {
public:
    virtual bool save(const CheckpointData& data) override;
    virtual bool load(const std::string& checkpoint_id, CheckpointData& data) override;
    virtual std::vector<CheckpointData> list(const std::string& graph_id) override;
    virtual bool remove(const std::string& checkpoint_id) override;
    virtual bool cleanup(const std::string& graph_id, size_t keep_count) override;

private:
    std::map<std::string, CheckpointData> checkpoints_;
    std::mutex mutex_;
};

// Redis 检查点实现
// 注意：需要 hiredis 库支持
class RedisCheckpoint : public Checkpoint {
public:
    RedisCheckpoint(const std::string& host = "127.0.0.1", int port = 6379,
                    const std::string& password = "");
    ~RedisCheckpoint();

    virtual bool save(const CheckpointData& data) override;
    virtual bool load(const std::string& checkpoint_id, CheckpointData& data) override;
    virtual std::vector<CheckpointData> list(const std::string& graph_id) override;
    virtual bool remove(const std::string& checkpoint_id) override;
    virtual bool cleanup(const std::string& graph_id, size_t keep_count) override;

    // 连接状态检查
    bool isConnected() const;
    bool reconnect();

private:
    void* redis_ctx_;  // redisContext* (使用 void* 避免头文件依赖)
    std::string host_;
    int port_;
    std::string password_;
    mutable std::mutex mutex_;
    int connection_timeout_ms_;

    std::string makeKey(const std::string& checkpoint_id);
    std::string makeListKey(const std::string& graph_id);
};

// GraphRunner 集成
class GraphRunner {
public:
    // 设置检查点管理器
    void setCheckpoint(Checkpoint* checkpoint);

    // 设置检查点配置
    void setCheckpointConfig(const CheckpointConfig& config);

    // 手动保存检查点
    bool saveCheckpoint();

    // 从检查点恢复
    bool loadCheckpoint(const std::string& checkpoint_id);

    // 获取检查点列表
    std::vector<CheckpointData> getCheckpoints();

protected:
    // 自动保存检查点（节点执行前/后）
    void autoSaveCheckpoint(Node* node, bool before_execution);

    // 恢复图状态
    bool restoreGraphState(const CheckpointData& data);

    // 恢复节点状态
    bool restoreNodeStates(const CheckpointData& data);

    // 恢复边数据
    bool restoreEdgeData(const CheckpointData& data);

private:
    Checkpoint* checkpoint_;
    CheckpointConfig checkpoint_config_;
    int64_t last_checkpoint_time_;
};

// Graph 扩展
class Graph {
public:
    // 获取当前状态（序列化）
    nlohmann::json getCurrentState();

    // 设置状态（反序列化）
    bool setState(const nlohmann::json& state);

    // 节点状态获取/设置
    nlohmann::json getNodeState(const std::string& node_name);
    void setNodeState(const std::string& node_name, const nlohmann::json& state);
};
```

### 核心操作流程

#### 保存流程
```
┌─────────────────┐
│  GraphRunner    │
│  执行节点       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  检查是否需要   │
│  保存检查点     │
└────────┬────────┘
         │ 是
         ▼
┌─────────────────┐
│  Graph          │
│  获取当前状态    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  收集节点状态    │
│  收集边数据      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Checkpoint     │
│  save()         │
└─────────────────┘
```

#### 恢复流程
```
┌─────────────────┐
│  Checkpoint     │
│  load()         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GraphRunner    │
│  restoreGraph   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  恢复节点状态    │
│  恢复边数据      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  继续执行       │
└─────────────────┘
```

### 技术细节
- 使用 nlohmann::json 进行序列化
- 检查点 ID 使用 UUID + 时间戳生成
- 支持压缩（可选，使用 zlib）
- Redis 存储使用 hash 结构

## 5. 实施步骤

### Step 1: 定义 Checkpoint 基类
- 设计 Checkpoint 接口
- 定义 CheckpointData 数据结构
- 定义 CheckpointConfig 配置结构
- 涉及文件：`framework/include/nndeploy/dag/checkpoint.h`

### Step 2: 实现 MemoryCheckpoint
- 实现内存存储的检查点
- 实现列表、删除、清理功能
- 涉及文件：`framework/include/nndeploy/dag/checkpoint.h`, `framework/source/nndeploy/dag/checkpoint.cc`

### Step 3: 实现 RedisCheckpoint
- 集成 Redis 客户端库（需要 hiredis 支持）
- 实现持久化存储
- 在 CMakeLists.txt 中添加 `ENABLE_NNDEPLOY_CHECKPOINT_REDIS` 编译选项
- 涉及文件：
  - `framework/include/nndeploy/dag/checkpoint.h`
  - `framework/source/nndeploy/dag/checkpoint.cc`
  - `cmake/checkpoint_redis.cmake` - 新增 Redis 检查和链接配置

### Step 4: 扩展 Graph 状态管理
- 实现 `getCurrentState()` 方法
- 实现 `setState()` 方法
- 涉及文件：`framework/include/nndeploy/dag/graph.h`, `framework/source/nndeploy/dag/graph.cc`

### Step 5: 集成 GraphRunner
- 实现 `setCheckpoint()` 和 `setCheckpointConfig()`
- 实现 `saveCheckpoint()` 和 `loadCheckpoint()`
- 实现自动保存逻辑
- 实现状态恢复逻辑
- 涉及文件：`framework/include/nndeploy/dag/graph_runner.h`, `framework/source/nndeploy/dag/graph_runner.cc`

### Step 6: 与人类反馈集成
- 在 CheckpointData 中添加反馈状态
- 保存待处理的反馈请求
- 恢复时加载反馈状态
- 涉及文件：`framework/include/nndeploy/dag/checkpoint.h`

### Step 7: 测试和文档
- 编写单元测试
- 编写使用示例
- 更新文档
- 涉及文件：`test/`, `docs/`

### 兼容性与迁移
- 检查点功能可选启用
- 不影响不使用检查点的现有代码
- 提供默认的 MemoryCheckpoint 实现

## 6. 验收标准

### 功能测试
- **测试用例 1**：使用 MemoryCheckpoint 保存和恢复执行状态
- **测试用例 2**：使用 RedisCheckpoint 保存和恢复执行状态
- **测试用例 3**：按节点执行前/后自动保存检查点
- **测试用例 4**：按时间间隔自动保存检查点
- **测试用例 5**：从检查点恢复后正确继续执行
- **测试用例 6**：检查点列表查询功能正确
- **测试用例 7**：清理旧检查点功能正确
- **测试用例 8**：与人类反馈机制配合，保存待处理反馈状态

### 代码质量
- 通过 GoogleTest 单元测试
- 代码覆盖率 > 80%
- 符合项目代码规范
- 头文件注释完善

### 回归测试
- 现有 DAG 执行功能正常运行
- 现有序列化/反序列化功能不受影响
- 各种 Executor 仍能正常工作

### 性能与可维护性
- 检查点保存时间与数据量成线性关系
- 不影响不使用检查点时的执行性能
- 代码结构清晰，易于扩展新的存储后端

### 文档与示例
- API 文档完整
- 提供使用示例
- 提供 Demo 程序

## 7. 其他说明

### 相关资源
- Redis 官方文档
- LangGraph checkpoint 文档

### 风险与应对
- **风险**：序列化复杂对象可能失败
  - **应对**：对不可序列化对象提供转换机制，充分测试
- **风险**：大检查点影响性能
  - **应对**：提供压缩选项，支持选择性保存
- **风险**：Redis 连接失败
  - **应对**：重试机制，降级到内存存储

### 依赖关系
- 依赖：无（核心功能独立）
- 协作：与人类反馈能力集成，保存待处理反馈状态
- 被依赖：时间旅行功能、人类反馈能力

### 扩展方向
- 支持文件系统存储
- 支持数据库存储（MySQL、PostgreSQL）
- 支持分布式检查点
- 支持检查点增量保存
