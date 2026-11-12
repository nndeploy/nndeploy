---
name: feature_human_feedback
title: 人类反馈能力（Human-in-the-Loop）
description: 支持DAG执行过程中暂停等待人类反馈，提供Approval、Correction、Skip、Reroute、Terminate五种反馈类型和断点续执行能力
category: [feature]
difficulty: medium
priority: P2
status: planned
version: 1.0.0
tags: [dag, human-in-the-loop, feedback, checkpoint, interrupt, resume]
estimated_time: 8h
files_affected: [framework/include/nndeploy/dag/human_feedback.h, framework/source/nndeploy/dag/human_feedback.cc, framework/include/nndeploy/dag/node.h, framework/include/nndeploy/dag/executor.h]
---

# Feature: 人类反馈能力（Human-in-the-Loop）

## 1. 背景（是什么 && 为什么）

### 现状分析
- **当前状态**：nndeploy DAG 模块具备基础的终止式中断机制，通过 `Node::interrupt()` 可以停止节点执行
- **存在问题**：当前中断机制只能终止执行，无法暂停等待人类输入后继续执行，无法实现人机协作工作流
- **需求痛点**：在实际应用中，许多场景需要人工介入进行审核、确认或修正，如内容审核、敏感操作确认、参数调整等

### 设计问题
- 当前 `Node` 状态机只包含 Idle、Running、Completed、Interrupted、Error 等状态，缺少 Waiting/Paused 状态
- 没有统一的反馈请求和响应管理机制
- Executor 层面缺乏反馈后的恢复执行逻辑
- 无法保存等待状态，导致中断后无法断点续执行

## 2. 目标（想做成什么样子）

### 核心目标
- **暂停等待能力**：节点执行过程中可暂停并等待人类反馈
- **反馈类型支持**：支持 Approval（批准）、Correction（修正）、Skip（跳过）、Reroute（重路由）、Terminate（终止）五种反馈类型
- **统一管理**：通过 HumanFeedbackManager 统一管理反馈请求和响应
- **断点续执行**：结合 Checkpoint 机制保存等待状态，支持断点恢复

### 预期效果
- 支持专门的 HumanReviewNode 审核节点
- 支持运行时动态请求人类反馈
- 提供完整的 API 供外部系统提交反馈
- 支持同步/异步两种反馈模式
- 超时控制和错误恢复机制

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要新增的文件
- `framework/include/nndeploy/dag/human_feedback.h` - 人类反馈相关类型定义和管理器
- `framework/source/nndeploy/dag/human_feedback.cc` - 反馈管理器实现

### 需要修改的文件
- `framework/include/nndeploy/dag/node.h` - 扩展状态机，添加反馈接口
- `framework/source/nndeploy/dag/node.cc` - 实现新增的状态和接口
- `framework/include/nndeploy/dag/executor.h` - 集成反馈恢复逻辑
- `framework/source/nndeploy/dag/executor.cc` - 实现 resumeNode 等方法
- `framework/include/nndeploy/dag/graph_runner.h` - 集成反馈流程控制
- `framework/include/nndeploy/dag/checkpoint.h` - 保存反馈请求状态（新增 checkpoint 时）

### 不能修改的文件
- `framework/include/nndeploy/dag/graph.h` - 图结构定义保持稳定，除非必要
- 公共 API 头文件中除新增接口外的已有接口

### 影响范围
- 所有 Executor 子类需要考虑反馈恢复
- 依赖中断机制的自定义节点可能需要适配

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：只能终止执行，无法恢复，状态不一致
- **新方案**：暂停-等待-继续流程，支持多种反馈类型，状态可恢复
- **核心变化**：扩展状态机、引入反馈管理器、Executor 支持恢复逻辑

### 架构/接口设计

#### 类型定义
```cpp
// 扩展节点状态
enum NodeState {
    kNodeStateIdle,          // 空闲
    kNodeStateRunning,       // 运行中
    kNodeStateWaiting,       // 等待人类反馈
    kNodeStatePaused,        // 已暂停
    kNodeStateCompleted,     // 完成
    kNodeStateInterrupted,   // 被终止
    kNodeStateError          // 错误
};

// 反馈类型
enum HumanFeedbackType {
    kFeedbackApproval,       // 批准，继续执行
    kFeedbackCorrection,     // 修正，带修改内容
    kFeedbackSkip,           // 跳过当前节点
    kFeedbackReroute,        // 重新路由到其他节点
    kFeedbackTerminate       // 终止整个流程
};

// 反馈请求
struct HumanFeedbackRequest {
    std::string request_id;           // 请求 ID
    std::string node_name;            // 请求节点
    std::string prompt;               // 提示语
    nlohmann::json context;           // 上下文数据
    nlohmann::json current_data;      // 当前节点数据
    std::vector<std::string> options; // 可选操作
    int64_t created_at;              // 创建时间
    int timeout_ms;                  // 超时时间
};

// 人类反馈
struct HumanFeedback {
    HumanFeedbackType type;          // 反馈类型
    std::string node_name;            // 关联节点
    std::string user_id;              // 用户标识
    nlohmann::json data;              // 反馈数据
    int64_t timestamp;                // 时间戳
    std::string session_id;           // 会话 ID
};
```

#### API 设计
```cpp
// Node 基类新增接口
class Node {
public:
    // 请求人类反馈（同步）
    virtual bool requestHumanFeedback(const std::string& prompt,
                                      const nlohmann::json& context,
                                      int timeout_ms = -1);

    // 请求人类反馈（异步）
    virtual bool requestHumanFeedbackAsync(const std::string& prompt,
                                           const nlohmann::json& context,
                                           std::function<void(const HumanFeedback&)> callback);

    // 接收人类反馈
    virtual void onHumanFeedback(const HumanFeedback& feedback);

    // 检查是否在等待反馈
    virtual bool isWaitingForFeedback() const;

    // 获取等待的反馈信息
    virtual HumanFeedbackRequest* getFeedbackRequest() const;
};

// 反馈管理器
class HumanFeedbackManager {
public:
    // 提交反馈请求
    std::string submitRequest(const HumanFeedbackRequest& request);

    // 提供反馈（外部调用）
    bool provideFeedback(const std::string& request_id,
                         const HumanFeedback& feedback);

    // 获取等待中的请求列表
    std::vector<HumanFeedbackRequest> getPendingRequests(const std::string& session_id);

    // 设置反馈回调
    void setFeedbackHandler(std::function<void(const HumanFeedback&)> handler);

    // 超时处理
    void handleTimeout(const std::string& request_id);
};

// 专门的审核节点
class HumanReviewNode : public Node {
public:
    HumanReviewNode(const std::string& name = "human_review");

    void setReviewOptions(const std::vector<std::string>& options);
    void setPromptTemplate(const std::string& template_str);

    virtual std::vector<Edge*> forward(std::vector<Edge*> inputs) override;
};

// 公共 API
namespace nndeploy {
namespace dag {
API_EXPORT bool submitHumanFeedback(const std::string& request_id,
                                     const HumanFeedback& feedback);
API_EXPORT std::vector<HumanFeedbackRequest> getPendingReviews(const std::string& session_id);
API_EXPORT void setGlobalFeedbackHandler(std::function<void(const HumanFeedback&)> handler);
}}
```

### 核心操作流程

```
┌─────────────┐
│   Node执行   │
└──────┬──────┘
       │ 检测需要人工审核
       ▼
┌──────────────────────┐
│ requestHumanFeedback │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  切换到Waiting状态    │
│  保存反馈请求        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Executor检测到Waiting │
│  保存Checkpoint      │
│  暂停执行            │
└──────┬───────────────┘
       │ 等待反馈
       ▼
┌──────────────────────┐
│  外部提供反馈        │
│  submitHumanFeedback │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  resumeNode()        │
│  根据反馈类型处理    │
└──────┬───────────────┘
       │
       ├── Approval → 继续执行
       ├── Correction → 应用修正后继续
       ├── Skip → 跳过当前节点
       ├── Reroute → 重新路由
       └── Terminate → 终止流程
```

### 技术细节
- 使用 `std::atomic<bool>` 和 `std::mutex` 保证线程安全
- 使用 `std::promise`/`std::future` 实现同步等待
- 反馈请求 ID 使用 UUID 生成
- 超时使用 `std::condition_variable::wait_for()` 实现

## 5. 实施步骤

### Step 1: 扩展 Node 状态机
- 在 `node.h` 中添加 `kNodeStateWaiting` 和 `kNodeStatePaused` 状态
- 添加反馈请求成员变量
- 添加 `requestHumanFeedback()` 等接口声明
- 涉及文件：`framework/include/nndeploy/dag/node.h`

### Step 2: 实现 Node 反馈接口
- 实现 `requestHumanFeedback()` 同步版本
- 实现 `requestHumanFeedbackAsync()` 异步版本
- 实现 `onHumanFeedback()` 响应处理
- 实现 `isWaitingForFeedback()` 状态检查
- 涉及文件：`framework/source/nndeploy/dag/node.cc`

### Step 3: 创建 HumanFeedbackManager
- 定义反馈管理器类
- 实现请求提交和反馈提供逻辑
- 实现超时处理机制
- 涉及文件：`framework/include/nndeploy/dag/human_feedback.h`, `framework/source/nndeploy/dag/human_feedback.cc`

### Step 4: 创建 HumanReviewNode
- 实现专用审核节点
- 支持审核选项配置
- 支持提示语模板
- 涉及文件：`framework/include/nndeploy/dag/human_feedback.h`

### Step 5: 扩展 Executor
- 添加 `setFeedbackManager()` 方法
- 实现 `resumeNode()` 方法
- 实现五种反馈类型的处理逻辑
- 涉及文件：`framework/include/nndeploy/dag/executor.h`, `framework/source/nndeploy/dag/executor.cc`

### Step 6: 集成 GraphRunner
- 在执行循环中检测 Waiting 状态
- 实现等待反馈逻辑
- 实现超时处理
- 涉及文件：`framework/include/nndeploy/dag/graph_runner.h`

### Step 7: 实现公共 API
- 导出 C API 函数
- 添加 Python 绑定
- 涉及文件：`framework/include/nndeploy/dag/human_feedback.h`, `python/` 相关文件

### 兼容性与迁移
- 新增功能不影响现有 API
- 默认行为保持不变（不请求反馈）
- 可选启用人类反馈功能

## 6. 验收标准

### 功能测试
- **测试用例 1**：创建 HumanReviewNode，配置审核选项，执行后能正确暂停等待反馈
- **测试用例 2**：提交 Approval 反馈，节点能继续执行完成
- **测试用例 3**：提交 Skip 反馈，节点被跳过，执行下一个节点
- **测试用例 4**：提交 Reroute 反馈，执行能跳转到指定节点
- **测试用例 5**：提交 Terminate 反馈，整个流程正确终止
- **测试用例 6**：超时无反馈时，流程能正确处理
- **测试用例 7**：异步反馈模式能正确工作
- **测试用例 8**：并发场景下反馈请求不会冲突

### 代码质量
- 通过 GoogleTest 单元测试
- 代码覆盖率 > 80%
- 符合项目代码规范
- 头文件注释完善

### 回归测试
- 现有 DAG 执行功能正常运行
- 现有中断机制不受影响
- 各种 Executor 仍能正常工作

### 性能与可维护性
- 反馈请求处理延迟 < 10ms
- 不影响正常图执行性能
- 代码结构清晰，易于扩展

### 文档与示例
- API 文档完整
- 提供使用示例
- 提供 Demo 程序

## 7. 其他说明

### 相关资源
- LangGraph interrupt 机制文档
- Issue：Human-in-the-Loop 支持

### 风险与应对
- **风险**：线程安全问题
  - **应对**：使用互斥锁保护共享状态，充分测试并发场景
- **风险**：死锁风险
  - **应对**：设置超时，避免无限等待
- **风险**：状态不一致
  - **应对**：设计清晰的状态转换规则，添加状态校验

### 依赖关系
- 依赖：无（核心功能独立）
- 被依赖：Checkpoint 机制（集成时保存反馈请求状态）
- 协作：与 Checkpoint 集成以支持反馈状态持久化
