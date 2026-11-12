---
name: feature_message_history
title: 消息历史管理
description: 提供统一的对话上下文管理能力，支持消息格式标准化、历史存储检索、窗口管理和持久化，适用于多轮对话LLM场景
category: [feature]
difficulty: easy
priority: P2
status: planned
version: 1.0.0
tags: [dag, message, llm, history, context, checkpoint]
estimated_time: 4h
files_affected: [framework/include/nndeploy/dag/message.h, framework/source/nndeploy/dag/message.cc, framework/include/nndeploy/dag/checkpoint.h]
---

# Feature: 消息历史管理

## 1. 背景（是什么 && 为什么）

### 现状分析
- **当前状态**：nndeploy DAG 没有内置的对话上下文管理能力
- **存在问题**：难以支持多轮对话场景，无法维护消息历史
- **需求痛点**：LLM 应用需要管理对话历史、上下文窗口、短期/长期记忆

### 设计问题
- 没有统一的消息格式定义
- 没有消息历史存储和检索机制
- 没有上下文窗口管理
- 无法与 Checkpoint 集成持久化

## 2. 目标（想做成什么样子）

### 核心目标
- **消息格式**：定义标准化的消息格式（role、content、metadata）
- **历史管理**：支持消息追加、截断、检索
- **窗口管理**：自动维护上下文窗口大小
- **持久化**：与 Checkpoint 集成，支持消息历史持久化

### 预期效果
- 提供统一的 Message 类
- 支持消息序列化/反序列化
- 支持按角色、时间、关键词检索
- 支持消息去重和合并

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要新增的文件
- `framework/include/nndeploy/dag/message.h` - 消息类和历史管理器
- `framework/source/nndeploy/dag/message.cc` - 消息实现

### 需要修改的文件
- `framework/include/nndeploy/dag/checkpoint.h` - 集成消息历史持久化
- `framework/include/nndeploy/dag/graph.h` - 添加消息历史管理（可选）

### 不能修改的文件
- 现有的数据传输接口保持兼容
- Edge、Executor 等模块保持不变

### 影响范围
- 主要影响 LLM 相关节点
- 与 Reducer 功能配合使用

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：无统一消息管理，各节点自行处理
- **新方案**：提供统一的消息类和历史管理器
- **核心变化**：新增 Message 和 MessageHistory 类

### 架构/接口设计

#### 类型定义
```cpp
// 消息角色
enum MessageRole {
    kRoleSystem,     // 系统消息
    kRoleUser,       // 用户消息
    kRoleAssistant,  // 助手消息
    kRoleTool,       // 工具消息
    kRoleCustom      // 自定义角色
};

// 消息类型
enum MessageType {
    kMessageTypeText,     // 文本消息
    kMessageTypeImage,    // 图像消息
    kMessageTypeAudio,    // 音频消息
    kMessageTypeVideo,    // 视频消息
    kMessageTypeToolCall, // 工具调用
    kMessageTypeMulti    // 多模态消息
};

// 消息元数据
struct MessageMetadata {
    int64_t timestamp;           // 时间戳
    std::string session_id;      // 会话 ID
    std::string user_id;         // 用户 ID
    std::string message_id;      // 消息唯一 ID
    int token_count;             // Token 数量（估算）
    std::map<std::string, std::string> extra;  // 额外元数据
};

// 消息类
class Message {
public:
    Message(MessageRole role, const std::string& content,
            MessageType type = kMessageTypeText);

    // 访问器
    MessageRole getRole() const { return role_; }
    void setRole(MessageRole role) { role_ = role; }

    std::string getContent() const { return content_; }
    void setContent(const std::string& content) { content_ = content; }

    MessageType getType() const { return type_; }
    void setType(MessageType type) { type_ = type; }

    const MessageMetadata& getMetadata() const { return metadata_; }
    MessageMetadata& getMetadata() { return metadata_; }

    // 序列化
    nlohmann::json toJson() const;
    static Message fromJson(const nlohmann::json& json);

    // 比较和哈希
    bool operator==(const Message& other) const;
    std::string getId() const;

private:
    MessageRole role_;
    std::string content_;
    MessageType type_;
    MessageMetadata metadata_;
};

// 消息历史管理器
class MessageHistory {
public:
    // Token 计数策略
    enum TokenCountStrategy {
        kTokenCountSimple,      // 简单估算：字符数 * 0.25
        kTokenCountAccurate,    // 精确计数（需要 tiktoken）
        kTokenCountExternal     // 外部传入（消息自带 token_count）
    };

    MessageHistory(const std::string& session_id = "",
                  TokenCountStrategy strategy = kTokenCountSimple);

    // 添加消息
    void addMessage(const Message& message);

    // 批量添加消息
    void addMessages(const std::vector<Message>& messages);

    // 获取所有消息
    std::vector<Message> getMessages() const;

    // 获取最近 N 条消息
    std::vector<Message> getLastNMessages(size_t n) const;

    // 按角色获取消息
    std::vector<Message> getMessagesByRole(MessageRole role) const;

    // 按时间范围获取消息
    std::vector<Message> getMessagesByTime(int64_t start, int64_t end) const;

    // 清空历史
    void clear();

    // 删除指定消息
    bool removeMessage(const std::string& message_id);

    // 获取 Token 总数（根据策略）
    int getTokenCount() const;

    // 设置 Token 计数策略
    void setTokenCountStrategy(TokenCountStrategy strategy);

    // 截断到指定 Token 数量
    void truncateToTokens(int max_tokens);

    // 截断到指定消息数
    void truncateToCount(size_t max_count);

    // 序列化
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& json);

    // 导出为 LLM API 格式
    nlohmann::json toLLMFormat() const;

    // 获取会话 ID
    std::string getSessionId() const;
    void setSessionId(const std::string& session_id);

private:
    void updateTokenCount(const Message& message);
    int estimateTokens(const std::string& text) const;

    std::vector<Message> messages_;
    std::string session_id_;
    int total_tokens_;
    TokenCountStrategy token_strategy_;
    std::mutex mutex_;
};

// 消息历史管理器（带持久化）
class MessageHistoryManager {
public:
    MessageHistoryManager(std::shared_ptr<Checkpoint> checkpoint);

    // 获取会话历史
    std::shared_ptr<MessageHistory> getHistory(const std::string& session_id);

    // 创建新会话
    std::shared_ptr<MessageHistory> createHistory(const std::string& session_id);

    // 删除会话
    bool deleteHistory(const std::string& session_id);

    // 获取所有会话列表
    std::vector<std::string> getSessions() const;

    // 持久化会话
    bool saveHistory(const std::string& session_id);

    // 加载会话
    bool loadHistory(const std::string& session_id);

private:
    std::map<std::string, std::shared_ptr<MessageHistory>> histories_;
    std::shared_ptr<Checkpoint> checkpoint_;
    std::mutex mutex_;
};
```

#### API 设计
```cpp
// 消息构建器（Builder 模式）
class MessageBuilder {
public:
    MessageBuilder& role(MessageRole role);
    MessageBuilder& content(const std::string& content);
    MessageBuilder& type(MessageType type);
    MessageBuilder& timestamp(int64_t timestamp);
    MessageBuilder& sessionId(const std::string& session_id);
    MessageBuilder& userId(const std::string& user_id);
    MessageBuilder& addMeta(const std::string& key, const std::string& value);
    MessageBuilder& tokenCount(int count);

    Message build();

private:
    MessageRole role_;
    std::string content_;
    MessageType type_;
    MessageMetadata metadata_;
};

// 消息工具函数
namespace nndeploy {
namespace dag {

// 创建常用消息
Message createSystemMessage(const std::string& content);
Message createUserMessage(const std::string& content);
Message createAssistantMessage(const std::string& content);
Message createToolMessage(const std::string& content, const std::string& tool_name);

// 消息格式转换
std::vector<nlohmann::json> toOpenAIMessages(const std::vector<Message>& messages);
std::vector<Message> fromOpenAIMessages(const std::vector<nlohmann::json>& messages);

// 消息合并（去重）
std::vector<Message> mergeMessages(const std::vector<Message>& a,
                                     const std::vector<Message>& b);

}} // namespace nndeploy::dag
```

### 核心操作流程

#### 消息添加流程
```
┌─────────────────┐
│  创建 Message    │
│  (role, content)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MessageHistory │
│  addMessage()   │
└────────┬────────┘
         │
         ├──▶ 添加到消息列表
         ├──▶ 更新 Token 计数
         └──▶ 触发持久化（可选）

         │
         ▼
┌─────────────────┐
│  消息已添加     │
└─────────────────┘
```

#### 检索流程
```
┌─────────────────┐
│  MessageHistory │
│  getMessages()  │
└────────┬────────┘
         │
         ├──▶ 全部消息
         ├──▶ 按 N 条
         ├──▶ 按角色
         └──▶ 按时间范围

         │
         ▼
┌─────────────────┐
│  返回消息列表    │
└─────────────────┘
```

### 技术细节
- 消息 ID 使用 UUID 生成
- Token 估算策略（可配置）：
  - 简单估算：字符数 * 0.25（默认）
  - 精确计数：集成 tiktoken（可选，需额外依赖）
  - 外部传入：由 LLM 模型返回精确计数
- 消息排序按时间戳
- 支持消息内容压缩（可选，使用 zlib）
- 消息去重策略：
  - 优先按 message_id 去重
  - 其次按 role + content 的哈希值去重
  - 可配置去重模式

## 5. 实施步骤

### Step 1: 定义消息类
- 定义 MessageRole、MessageType 枚举
- 定义 MessageMetadata 结构
- 实现 Message 类
- 涉及文件：`framework/include/nndeploy/dag/message.h`, `framework/source/nndeploy/dag/message.cc`

### Step 2: 实现消息历史管理器
- 实现 MessageHistory 类
- 实现消息添加、检索、截断
- 实现序列化/反序列化
- 涉及文件：`framework/include/nndeploy/dag/message.h`, `framework/source/nndeploy/dag/message.cc`

### Step 3: 实现带持久化的管理器
- 实现 MessageHistoryManager 类
- 与 Checkpoint 集成
- 涉及文件：`framework/include/nndeploy/dag/message.h`, `framework/source/nndeploy/dag/message.cc`

### Step 4: 实现工具函数和构建器
- 实现 MessageBuilder
- 实现常用消息创建函数
- 实现格式转换函数
- 涉及文件：`framework/include/nndeploy/dag/message.h`, `framework/source/nndeploy/dag/message.cc`

### Step 5: 集成 Reducer
- 使用 AddMessagesReducer 管理消息历史
- 涉及文件：`framework/include/nndeploy/dag/reducer.h`

### Step 6: 测试和文档
- 编写单元测试
- 编写使用示例
- 编写 LLM 集成示例
- 涉及文件：`test/`, `docs/`

### 兼容性与迁移
- 消息功能可选使用
- 不影响现有 DAG 执行
- 现有代码无需修改

## 6. 验收标准

### 功能测试
- **测试用例 1**：正确创建和添加消息
- **测试用例 2**：检索最近 N 条消息正确
- **测试用例 3**：按角色检索消息正确
- **测试用例 4**：Token 截断功能正确
- **测试用例 5**：消息序列化/反序列化正确
- **测试用例 6**：与 Checkpoint 集成持久化正确
- **测试用例 7**：消息去重和合并正确
- **测试用例 8**：LLM 格式转换正确

### 代码质量
- 通过 GoogleTest 单元测试
- 代码覆盖率 > 80%
- 符合项目代码规范
- 头文件注释完善

### 回归测试
- 现有 DAG 执行功能正常运行
- 不使用消息历史时无影响
- 各种 Executor 仍能正常工作

### 性能与可维护性
- 消息添加延迟 < 1ms
- 消息检索延迟 < 10ms
- 代码结构清晰，易于扩展

### 文档与示例
- API 文档完整
- 提供使用示例
- 提供 LLM 集成示例

## 7. 其他说明

### 相关资源
- OpenAI Chat API 格式
- LangChain Message 文档

### 风险与应对
- **风险**：Token 估算不准确
  - **应对**：支持外部 Token 计数器，提供精确计数选项（tiktoken 集成）
- **风险**：消息历史过大导致性能问题
  - **应对**：自动截断机制，提供手动清理 API，支持消息摘要压缩
- **风险**：多线程并发问题
  - **应对**：使用互斥锁保护消息列表，提供线程安全的批量操作
- **风险**：消息去重不准确
  - **应对**：多级去重策略（ID、哈希），可配置去重模式

### 依赖关系
- 依赖：Reducer（AddMessagesReducer）
- 依赖：Checkpoint（持久化）
- 被依赖：LLM 插件

### 扩展方向
- 支持消息向量检索（语义搜索）
- 支持消息摘要和压缩
- 支持多模态消息处理
- 支持消息加密存储
