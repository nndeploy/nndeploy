---
name: feature_time_travel
title: 时间旅行（Time Travel）
description: 支持图执行状态的版本管理和历史追踪，可以回退到任意检查点、创建分支执行和比较版本差异
category: [feature]
difficulty: hard
priority: P3
status: planned
version: 1.0.0
tags: [dag, time-travel, checkpoint, version, branch, history]
estimated_time: 16h
files_affected: [framework/include/nndeploy/dag/time_travel.h, framework/source/nndeploy/dag/time_travel.cc, framework/include/nndeploy/dag/checkpoint.h, framework/include/nndeploy/dag/graph_runner.h]
---

# Feature: 时间旅行（Time Travel）

## 1. 背景（是什么 && 为什么）

### 现状分析
- **当前状态**：nndeploy DAG 执行完成后无法回溯和修改执行路径
- **存在问题**：调试能力有限，无法追溯历史状态和分支执行
- **需求痛点**：需要回退到任意检查点、修改参数后继续执行、探索不同执行路径

### 设计问题
- 无法保存完整的执行历史
- 没有状态版本管理
- 不支持回退和分支执行
- 无法比较不同执行路径的结果

## 2. 目标（想做成什么样子）

### 核心目标
- **历史追踪**：保存完整的执行历史和检查点版本
- **状态回退**：回退到任意历史检查点
- **分支执行**：从历史检查点创建分支，探索不同路径
- **版本管理**：支持检查点版本标记和比较

### 预期效果
- 用户可以查看执行历史时间线
- 可以回退到任意检查点
- 可以从历史点创建新分支执行
- 可以比较不同版本的状态差异

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要新增的文件
- `framework/include/nndeploy/dag/time_travel.h` - 时间旅行接口
- `framework/source/nndeploy/dag/time_travel.cc` - 时间旅行实现

### 需要修改的文件
- `framework/include/nndeploy/dag/checkpoint.h` - 扩展历史记录功能
- `framework/source/nndeploy/dag/checkpoint.cc` - 实现历史管理
- `framework/include/nndeploy/dag/graph_runner.h` - 集成时间旅行

### 不能修改的文件
- 现有的检查点基本接口保持兼容
- 图结构定义保持不变

### 影响范围
- Checkpoint 存储需求增加
- 需要考虑历史数据的清理策略

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：检查点只保存当前状态，无历史追踪
- **新方案**：保存完整历史，支持时间旅行和分支执行
- **核心变化**：扩展 Checkpoint 系统，新增 TimeTravel 接口

### 架构/接口设计

#### 类型定义
```cpp
// 检查点版本信息
struct CheckpointVersion {
    std::string version_id;        // 版本 ID
    std::string parent_id;         // 父版本 ID
    std::string checkpoint_id;     // 关联的检查点 ID
    std::string name;              // 版本名称（可选）
    std::string description;       // 版本描述
    int64_t created_at;           // 创建时间
    std::vector<std::string> tags; // 标签
};

// 时间线节点
struct TimelineNode {
    CheckpointVersion version;
    std::vector<TimelineNode> children;  // 子分支
};

// 时间旅行操作
enum TimeTravelAction {
    kTravelRestore,       // 恢复到指定版本
    kTravelBranch,        // 从指定版本创建分支
    kTravelDelete,        // 删除版本及其子版本
    kTravelTag            // 添加/删除标签
};

// 时间旅行接口
class TimeTravel {
public:
    virtual ~TimeTravel() = default;

    // 获取完整时间线
    virtual TimelineNode getTimeline(const std::string& root_id) = 0;

    // 创建版本快照
    virtual std::string createVersion(const std::string& checkpoint_id,
                                       const std::string& name = "",
                                       const std::string& description = "") = 0;

    // 恢复到指定版本
    virtual bool restoreVersion(const std::string& version_id) = 0;

    // 从版本创建分支
    virtual std::string createBranch(const std::string& parent_version_id,
                                      const std::string& branch_name) = 0;

    // 删除版本（及其子版本）
    virtual bool deleteVersion(const std::string& version_id) = 0;

    // 比较两个版本的差异
    virtual nlohmann::json diffVersions(const std::string& version_id1,
                                         const std::string& version_id2) = 0;

    // 查找版本
    virtual CheckpointVersion findVersion(const std::string& version_id) = 0;
    virtual std::vector<CheckpointVersion> findVersionsByTag(const std::string& tag) = 0;

    // 获取版本列表
    virtual std::vector<CheckpointVersion> getVersions(const std::string& root_id) = 0;
};

// 内存时间旅行实现
class MemoryTimeTravel : public TimeTravel {
public:
    MemoryTimeTravel(std::shared_ptr<Checkpoint> checkpoint);

    virtual TimelineNode getTimeline(const std::string& root_id) override;
    virtual std::string createVersion(const std::string& checkpoint_id,
                                       const std::string& name = "",
                                       const std::string& description = "") override;
    virtual bool restoreVersion(const std::string& version_id) override;
    virtual std::string createBranch(const std::string& parent_version_id,
                                      const std::string& branch_name) override;
    virtual bool deleteVersion(const std::string& version_id) override;
    virtual nlohmann::json diffVersions(const std::string& version_id1,
                                         const std::string& version_id2) override;
    virtual CheckpointVersion findVersion(const std::string& version_id) override;
    virtual std::vector<CheckpointVersion> findVersionsByTag(const std::string& tag) override;
    virtual std::vector<CheckpointVersion> getVersions(const std::string& root_id) override;

private:
    TimelineNode buildTimeline(const std::string& version_id);
    std::shared_ptr<Checkpoint> checkpoint_;
    std::map<std::string, CheckpointVersion> versions_;
    std::map<std::string, std::vector<std::string>> children_map_;
    std::mutex mutex_;
};

// Checkpoint 扩展
class Checkpoint {
public:
    // 扩展：设置时间旅行
    void setTimeTravel(std::shared_ptr<TimeTravel> time_travel);

    // 扩展：自动创建版本（可选）
    void enableAutoVersioning(bool enable);

    // 扩展：创建当前状态的版本
    std::string createCurrentVersion(const std::string& name = "",
                                      const std::string& description = "");

protected:
    std::shared_ptr<TimeTravel> time_travel_;
    bool auto_versioning_;
};

// GraphRunner 集成
class GraphRunner {
public:
    // 设置时间旅行管理器
    void setTimeTravel(std::shared_ptr<TimeTravel> time_travel);

    // 创建当前状态的版本
    std::string createVersion(const std::string& name = "",
                              const std::string& description = "");

    // 恢复到指定版本
    bool restoreVersion(const std::string& version_id);

    // 从版本创建分支
    std::string createBranch(const std::string& version_id,
                              const std::string& branch_name);

    // 获取执行历史时间线
    TimelineNode getTimeline();

    // 比较版本差异
    nlohmann::json diffVersions(const std::string& version_id1,
                                 const std::string& version_id2);

protected:
    std::shared_ptr<TimeTravel> time_travel_;
    std::string root_version_id_;  // 当前执行根版本
};
```

### 核心操作流程

#### 版本创建流程
```
┌─────────────────┐
│  GraphRunner    │
│  执行完成       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  createVersion()│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Checkpoint     │
│  save()         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TimeTravel     │
│  createVersion()│
│  记录版本信息    │
└─────────────────┘
```

#### 分支执行流程
```
┌─────────────────┐       ┌─────────────────┐
│  版本 A         │       │  版本 B         │
│  (历史)         │──────▶│  (分支)         │
└────────┬────────┘       └────────┬────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐       ┌─────────────────┐
│  恢复 A 状态    │       │  从 A 恢复      │
│  继续执行       │       │  修改后执行     │
└─────────────────┘       └─────────────────┘
```

#### 时间线可视化
```
Root (v1)
├── v2
│   ├── v3 (当前)
│   └── v4 (分支)
└── v5
    └── v6 (实验分支)
```

### 技术细节
- 版本 ID 使用 UUID
- 支持版本树形结构（类似 Git）
- 版本差异使用 JSON diff 算法（支持 patch 应用）
- 支持版本标签过滤
- 版本数据压缩（可选，使用 zlib）
- 版本回收策略：
  - 保留最近 N 个版本
  - 按时间窗口清理
  - 标记为 "keep" 的版本不自动删除

## 5. 实施步骤

### Step 1: 定义时间旅行接口
- 定义 CheckpointVersion、TimelineNode 结构
- 定义 TimeTravel 接口
- 涉及文件：`framework/include/nndeploy/dag/time_travel.h`

### Step 2: 实现 MemoryTimeTravel
- 实现版本管理逻辑
- 实现时间线构建
- 实现版本差异比较
- 涉及文件：`framework/include/nndeploy/dag/time_travel.h`, `framework/source/nndeploy/dag/time_travel.cc`

### Step 3: 扩展 Checkpoint
- 添加 TimeTravel 集成
- 实现自动版本化
- 涉及文件：`framework/include/nndeploy/dag/checkpoint.h`, `framework/source/nndeploy/dag/checkpoint.cc`

### Step 4: 集成 GraphRunner
- 实现版本创建/恢复接口
- 实现分支执行逻辑
- 涉及文件：`framework/include/nndeploy/dag/graph_runner.h`, `framework/source/nndeploy/dag/graph_runner.cc`

### Step 5: 实现 JSON diff 算法
- 实现差异比较算法
- 支持差异可视化
- 涉及文件：
  - `framework/include/nndeploy/dag/json_diff.h` - 新增 JSON diff 类型定义
  - `framework/source/nndeploy/dag/json_diff.cc` - diff 算法实现
  - `framework/source/nndeploy/dag/time_travel.cc`

### Step 6: 测试和文档
- 编写单元测试
- 编写使用示例
- 编写可视化工具
- 涉及文件：`test/`, `docs/`, `demo/`

### 兼容性与迁移
- 时间旅行功能可选启用
- 不启用时无性能影响
- 现有代码无需修改

## 6. 验收标准

### 功能测试
- **测试用例 1**：创建版本并正确记录历史
- **测试用例 2**：恢复到指定版本状态正确
- **测试用例 3**：从版本创建分支并执行
- **测试用例 4**：获取完整时间线正确
- **测试用例 5**：版本差异比较正确
- **测试用例 6**：删除版本及其子版本正确
- **测试用例 7**：按标签查找版本正确
- **测试用例 8**：分支执行不影响主分支

### 代码质量
- 通过 GoogleTest 单元测试
- 代码覆盖率 > 80%
- 符合项目代码规范
- 头文件注释完善

### 回归测试
- 现有 DAG 执行功能正常运行
- 现有 Checkpoint 功能不受影响
- 不使用时间旅行时无性能退化

### 性能与可维护性
- 版本创建延迟 < 100ms
- 版本恢复延迟 < 50ms
- 代码结构清晰，易于扩展

### 文档与示例
- API 文档完整
- 提供使用示例
- 提供时间线可视化工具

## 7. 其他说明

### 相关资源
- Git 版本管理概念
- JSON diff 算法

### 风险与应对
- **风险**：历史数据存储占用空间大
  - **应对**：提供清理策略，支持压缩，支持增量存储
- **风险**：版本树过深影响性能
  - **应对**：优化查询算法，提供分支合并，设置最大深度限制
- **风险**：并发修改冲突
  - **应对**：乐观并发控制，冲突检测，提供冲突解决 API
- **风险**：版本差异计算复杂度高
  - **应对**：使用高效的 diff 算法，支持选择性差异计算

### 依赖关系
- 依赖：Checkpoint（基础）
- 被依赖：调试工具

### 扩展方向
- 支持 Redis/分布式时间旅行
- 支持版本合并（merge）
- 支持回滚操作（rollback）
- 支持版本导出/导入
