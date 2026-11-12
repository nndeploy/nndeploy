---
name: feature_dynamic_graph
title: 动态图修改
description: 支持运行时动态添加/删除节点和边，提供事务性修改、快照回滚和修改历史追踪能力
category: [feature]
difficulty: hard
priority: P2
status: planned
version: 1.0.0
tags: [dag, graph, dynamic, transaction, rollback, snapshot]
estimated_time: 16h
files_affected: [framework/include/nndeploy/dag/graph.h, framework/source/nndeploy/dag/graph.cc, framework/include/nndeploy/dag/graph_transaction.h, framework/source/nndeploy/dag/graph_transaction.cc]
---

# Feature: 动态图修改

## 1. 背景（是什么 && 为什么）

### 现状分析
- **当前状态**：nndeploy DAG 图结构在运行时不可修改
- **存在问题**：无法动态添加/删除节点，无法修改边连接
- **需求痛点**：需要运行时调整图结构、热更新能力、动态分支切换

### 设计问题
- Graph 类缺少动态修改 API
- 图结构修改后需要重新编译
- 缺少修改后的验证机制
- 没有回滚机制

## 2. 目标（想做成什么样子）

### 核心目标
- **动态修改**：运行时添加/删除节点和边
- **热更新**：修改后无需重启即可生效
- **验证机制**：修改后验证图的有效性
- **回滚机制**：支持修改回滚

### 预期效果
- 用户可以在运行时修改图结构
- 支持批量修改操作
- 支持事务性修改（原子性）
- 支持修改历史和回滚

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要修改的文件
- `framework/include/nndeploy/dag/graph.h` - 添加动态修改 API
- `framework/source/nndeploy/dag/graph.cc` - 实现动态修改逻辑

### 需要新增的文件
- `framework/include/nndeploy/dag/graph_transaction.h` - 事务性修改支持
- `framework/source/nndeploy/dag/graph_transaction.cc` - 事务实现

### 不能修改的文件
- 现有的图序列化/反序列化保持兼容
- 基础图结构类保持稳定

### 影响范围
- 执行中的图可能需要暂停才能修改
- 需要考虑线程安全问题

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：图结构在编译时确定，运行时不可修改
- **新方案**：支持运行时动态修改图结构
- **核心变化**：Graph 新增动态修改 API，引入事务机制

### 架构/接口设计

#### 类型定义
```cpp
// 图修改操作类型
enum GraphOperationType {
    kOpAddNode,           // 添加节点
    kOpRemoveNode,        // 删除节点
    kOpAddEdge,           // 添加边
    kOpRemoveEdge,        // 删除边
    kOpModifyNode,        // 修改节点参数
    kOpBatch              // 批量操作
};

// 图修改操作
struct GraphOperation {
    GraphOperationType type;
    nlohmann::json params;

    // 节点操作
    std::string node_name;
    std::string node_type;
    nlohmann::json node_params;

    // 边操作
    std::string source_node;
    std::string target_node;
    std::string edge_name;
    size_t source_port;
    size_t target_port;

    // 元数据
    int64_t timestamp;
    std::string operation_id;
};

// 图验证结果
struct GraphValidationResult {
    bool valid;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    bool has_cycles;          // 是否有环（允许则通过）
    bool all_nodes_connected; // 所有节点是否可达
};

// 图事务
class GraphTransaction {
public:
    GraphTransaction(Graph* graph);

    // 添加节点
    bool addNode(const std::string& name,
                  const std::string& type,
                  const nlohmann::json& params = {});

    // 删除节点
    bool removeNode(const std::string& name);

    // 添加边
    bool addEdge(const std::string& source,
                  const std::string& target,
                  const std::string& edge_name = "",
                  size_t source_port = 0,
                  size_t target_port = 0);

    // 删除边
    bool removeEdge(const std::string& edge_name);

    // 修改节点参数
    bool modifyNode(const std::string& name, const nlohmann::json& params);

    // 提交事务
    bool commit();

    // 回滚事务
    bool rollback();

    // 验证图
    GraphValidationResult validate();

    // 获取操作列表
    const std::vector<GraphOperation>& getOperations() const { return operations_; }

    // 清空操作
    void clear();

private:
    Graph* graph_;
    std::vector<GraphOperation> operations_;
    std::map<std::string, Node*> added_nodes_;
    std::map<std::string, Edge*> added_edges_;
    std::vector<Node*> removed_nodes_;
    std::vector<Edge*> removed_edges_;
    std::map<std::string, nlohmann::json> original_params_;
};

// Graph 扩展
class Graph : public Node {
public:
    // ========== 动态修改 API ==========

    // 添加节点
    bool addNode(const std::string& name,
                  const std::string& type,
                  const nlohmann::json& params = {});

    // 删除节点
    bool removeNode(const std::string& name);

    // 添加边
    bool addEdge(const std::string& source,
                  const std::string& target,
                  const std::string& edge_name = "",
                  size_t source_port = 0,
                  size_t target_port = 0);

    // 删除边
    bool removeEdge(const std::string& edge_name);

    // 修改节点参数
    bool modifyNode(const std::string& name, const nlohmann::json& params);

    // 获取节点
    Node* getNode(const std::string& name);
    const Node* getNode(const std::string& name) const;

    // 获取边
    Edge* getEdge(const std::string& name);
    const Edge* getEdge(const std::string& name) const;

    // 获取所有节点
    std::vector<Node*> getNodes();

    // 获取所有边
    std::vector<Edge*> getEdges();

    // 验证图
    GraphValidationResult validate(bool check_cycles = true) const;

    // 重建拓扑排序
    bool rebuildTopology();

    // ========== 事务支持 ==========

    // 创建事务
    std::shared_ptr<GraphTransaction> createTransaction();

    // 执行事务
    bool executeTransaction(const std::shared_ptr<GraphTransaction>& transaction);

    // ========== 快照和回滚 ==========

    // 创建快照
    std::string createSnapshot();

    // 恢复快照
    bool restoreSnapshot(const std::string& snapshot_id);

    // 删除快照
    bool deleteSnapshot(const std::string& snapshot_id);

    // 获取快照列表
    std::vector<std::string> getSnapshots();

    // ========== 修改历史 ==========

    // 获取修改历史
    const std::vector<GraphOperation>& getModificationHistory() const;

    // 清空修改历史
    void clearModificationHistory();

    // 启用/禁用修改追踪
    void setModifyTracking(bool enable);
    bool isModifyTrackingEnabled() const;

protected:
    // 修改锁（用于并发控制）
    std::shared_mutex modify_lock_;

    // 修改历史
    std::vector<GraphOperation> modification_history_;
    bool modify_tracking_enabled_;

    // 快照管理
    struct GraphSnapshot {
        std::string snapshot_id;
        std::string graph_json;
        int64_t timestamp;
    };
    std::map<std::string, GraphSnapshot> snapshots_;
    std::mutex snapshots_mutex_;
};
```

#### API 设计
```cpp
// 节点工厂（用于动态创建）
class NodeFactory {
public:
    // 注册节点类型
    static void registerNode(const std::string& type,
                             std::function<Node*()> creator);

    // 创建节点
    static Node* createNode(const std::string& type,
                             const nlohmann::json& params = {});

    // 检查类型是否注册
    static bool hasType(const std::string& type);

    // 获取所有注册类型
    static std::vector<std::string> getTypes();

private:
    static std::map<std::string, std::function<Node*()>> creators_;
    static std::mutex mutex_;
};

// 便捷操作宏
#define NNDEPLOY_REGISTER_NODE_TYPE(type, class_name) \
    namespace { \
        struct Register_##class_name { \
            Register_##class_name() { \
                nndeploy::dag::NodeFactory::registerNode( \
                    type, []() -> nndeploy::dag::Node* { return new class_name(); }); \
            } \
        } register_##class_name##_instance; \
    }

// 便捷事务使用
#define NNDEPLOY_GRAPH_TRANSACTION(graph, code) \
    do { \
        auto transaction = (graph)->createTransaction(); \
        { code } \
        if (transaction.validate().valid) { \
            if (!transaction.commit()) { \
                transaction.rollback(); \
            } \
        } \
    } while(0)
```

### 核心操作流程

#### 事务修改流程
```
┌─────────────────┐
│  创建事务       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  执行修改操作   │
│  (暂不应用)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  验证图         │
└────────┬────────┘
         │
         ├──▶ 通过 ──▶ 提交 ──▶ 生效
         │
         └──▶ 失败 ──▶ 回滚
```

#### 快照和恢复流程
```
┌─────────────────┐
│  createSnapshot │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  序列化图       │
│  保存快照       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  执行修改...    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  restoreSnapshot│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  反序列化       │
│  恢复图         │
└─────────────────┘
```

### 技术细节
- 使用读写锁保护图结构（std::shared_mutex）
- 快照使用 JSON 序列化（可选压缩）
- 拓扑排序在修改后重建（使用 Kahn 算法）
- 支持增量验证（只验证受影响的子图）
- 并发修改策略：
  - 写操作互斥（使用互斥锁）
  - 读操作可并发（使用共享锁）
  - 执行中的图修改需要暂停（提供暂停/恢复 API）
- 事务隔离级别：READ_COMMITTED（提交后才对其他操作可见）

## 5. 实施步骤

### Step 1: 定义图修改类型
- 定义 GraphOperationType、GraphOperation 结构
- 定义 GraphValidationResult 结构
- 涉及文件：`framework/include/nndeploy/dag/graph_transaction.h`

### Step 2: 实现 GraphTransaction
- 实现各种图修改操作
- 实现事务提交和回滚
- 实现图验证逻辑
- 涉及文件：`framework/include/nndeploy/dag/graph_transaction.h`, `framework/source/nndeploy/dag/graph_transaction.cc`

### Step 3: 扩展 Graph 类
- 添加动态修改 API
- 实现快照和恢复
- 实现修改历史追踪
- 涉及文件：`framework/include/nndeploy/dag/graph.h`, `framework/source/nndeploy/dag/graph.cc`

### Step 4: 实现 NodeFactory
- 实现节点注册和创建
- 涉及文件：
  - `framework/include/nndeploy/dag/node_factory.h` - 新增节点工厂类
  - `framework/source/nndeploy/dag/node_factory.cc` - 节点工厂实现

### Step 5: 集成 GraphRunner
- 支持运行时修改检测
- 实现热更新逻辑
- 涉及文件：`framework/include/nndeploy/dag/graph_runner.h`, `framework/source/nndeploy/dag/graph_runner.cc`

### Step 6: 测试和文档
- 编写单元测试
- 编写使用示例
- 编写动态修改 Demo
- 涉及文件：`test/`, `docs/`, `demo/`

### 兼容性与迁移
- 动态修改功能可选使用
- 不修改图时行为不变
- 现有代码无需修改

## 6. 验收标准

### 功能测试
- **测试用例 1**：运行时添加节点正确
- **测试用例 2**：运行时删除节点正确
- **测试用例 3**：运行时添加/删除边正确
- **测试用例 4**：事务提交正确应用修改
- **测试用例 5**：事务回滚正确撤销修改
- **测试用例 6**：图验证正确检测错误
- **测试用例 7**：快照保存和恢复正确
- **测试用例 8**：修改历史记录正确

### 代码质量
- 通过 GoogleTest 单元测试
- 代码覆盖率 > 80%
- 符合项目代码规范
- 头文件注释完善

### 回归测试
- 现有 DAG 执行功能正常运行
- 不使用动态修改时无影响
- 现有图定义和加载功能正常

### 性能与可维护性
- 添加节点延迟 < 10ms
- 拓扑排序重建延迟 < 50ms
- 代码结构清晰，易于扩展

### 文档与示例
- API 文档完整
- 提供使用示例
- 提供动态修改 Demo

## 7. 其他说明

### 相关资源
- 图数据结构
- 事务处理概念

### 风险与应对
- **风险**：并发修改导致数据不一致
  - **应对**：使用读写锁，提供事务隔离，乐观并发控制
- **风险**：修改导致执行中的图崩溃
  - **应对**：运行时修改需要暂停，提供检查点恢复，修改前验证
- **风险**：内存泄漏（节点未正确删除）
  - **应对**：使用智能指针，提供清理 API，定期泄漏检测
- **风险**：拓扑排序失败（环检测）
  - **应对**：修改前验证无环，提供环检测和报告
- **风险**：快照占用大量内存
  - **应对**：支持增量快照，限制快照数量，自动清理

### 依赖关系
- 依赖：图基础结构
- 被依赖：事件驱动

### 扩展方向
- 支持图差异计算和可视化
- 支持图迁移（版本升级）
- 支持图模板和实例化
- 支持图的分布式修改
