---
name: feature_node_skill_tool_doc
title: 节点 Skill、Tool、Doc 能力增强
description: 为DAG节点添加结构化的能力描述（Skill）、外部工具接口（Tool）和文档元数据（Doc），支持自动化发现和调用
category: [feature]
difficulty: medium
priority: P3
status: planned
version: 1.0.0
tags: [dag, node, skill, tool, doc, registry, agent]
estimated_time: 8h
files_affected: [framework/include/nndeploy/dag/node.h, framework/source/nndeploy/dag/node.cc, framework/include/nndeploy/dag/graph.h, framework/source/nndeploy/dag/graph.cc, framework/include/nndeploy/dag/registry.h]
---

# Feature: 节点 Skill、Tool、Doc 能力增强

## 1. 背景（是什么 && 为什么）

### 现状分析
- 当前 nndeploy 的 DAG 节点（Node）主要用于执行固定的推理和处理流程
- 节点缺乏对外暴露能力的描述机制（Skill）
- 节点缺少与外部工具交互的标准化接口（Tool）
- 节点文档散落在代码注释中，没有结构化的元数据（Doc）
- 这导致节点难以被自动化工具发现、调用和理解

### 设计问题
- **具体的技术问题**: 节点能力描述缺失，需要手动查阅源码才能了解节点功能
- **架构层面的不足**: 节点设计过于封闭，缺乏与 Agent/AI 系统集成的标准接口
- **用户体验的缺陷**: 难以通过程序化方式发现和使用节点能力

## 2. 目标（想做成什么样子）

### 核心目标
- 为节点添加 **Skill** 描述机制：声明节点具备的能力和参数
- 为节点添加 **Tool** 接口：支持调用外部工具/服务
- 为节点添加 **Doc** 元数据：结构化的节点文档
- 提供标准化的注册和查询接口

### 预期效果
- **功能层面的改进**: 节点能力可被自动化发现和调用
- **性能层面的提升**: 通过结构化元数据支持更高效的节点查找和组合
- **用户体验的优化**:
  - Agent/AI 系统能理解节点能力并自主调用
  - 用户可通过 API 查询节点文档
  - 节点可被自动化的工作流编排工具使用

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要修改的文件
- `framework/include/nndeploy/dag/node.h` - 添加 Skill、Tool、Doc 相关接口
- `framework/source/nndeploy/dag/node.cc` - 实现相关功能
- `framework/include/nndeploy/dag/graph.h` - 添加节点查询接口
- `framework/source/nndeploy/dag/graph.cc` - 实现节点查询
- `plugin/source/nndeploy/*/config.cmake` - 更新各插件的注册逻辑

### 需要新增的文件
- `framework/include/nndeploy/dag/node_skill.h` - Skill 定义
- `framework/include/nndeploy/dag/node_tool.h` - Tool 定义
- `framework/include/nndeploy/dag/node_doc.h` - Doc 定义
- `framework/source/nndeploy/dag/node_skill.cc` - Skill 实现
- `framework/source/nndeploy/dag/node_tool.cc` - Tool 实现
- `framework/source/nndeploy/dag/node_doc.cc` - Doc 实现
- `framework/include/nndeploy/dag/registry.h` - 节点能力注册中心

### 不能修改的文件
- 现有插件的 core 实现文件 - 保持向后兼容，新能力为可选特性

### 影响范围
- 所有 DAG 节点类型
- 与 Agent/AI 系统的集成
- 工作流可视化工具

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**: 节点能力隐含在代码中，需要手动阅读源码
- **新方案**: 通过 Skill、Tool、Doc 三个维度结构化描述节点能力
- **核心变化**: 引入元数据驱动的节点能力发现机制

### 架构/接口设计

#### 1. Skill - 节点能力描述
```cpp
// framework/include/nndeploy/dag/node_skill.h

namespace nndeploy {
namespace dag {

// 技能参数定义
struct SkillParam {
  std::string name;
  std::string type;  // "string", "int", "float", "tensor", "image", etc.
  std::string description;
  bool required = false;
  std::any default_value;

  // 验证函数
  std::function<bool(const std::any&)> validator;
};

// 节点技能定义
class NNDEPLOY_CC_API NodeSkill {
 public:
  NodeSkill(const std::string& name, const std::string& description);
  virtual ~NodeSkill() = default;

  // 添加参数
  void addParam(const SkillParam& param);
  void addParam(const std::string& name, const std::string& type,
               const std::string& description, bool required = false);

  // 获取技能信息
  const std::string& getName() const { return name_; }
  const std::string& getDescription() const { return description_; }
  const std::vector<SkillParam>& getParams() const { return params_; }

  // 验证参数
  bool validateParams(const std::map<std::string, std::any>& params) const;

 private:
  std::string name_;
  std::string description_;
  std::vector<SkillParam> params_;
};

}  // namespace dag
}  // namespace nndeploy
```

#### 2. Tool - 外部工具接口
```cpp
// framework/include/nndeploy/dag/node_tool.h

namespace nndeploy {
namespace dag {

// 工具调用结果
struct ToolResult {
  base::Status status;
  std::string output;
  std::map<std::string, std::any> data;
};

// 工具定义
class NNDEPLOY_CC_API NodeTool {
 public:
  NodeTool(const std::string& name, const std::string& description);
  virtual ~NodeTool() = default;

  // 执行工具调用
  virtual ToolResult execute(const std::map<std::string, std::any>& params) = 0;

  // 获取工具信息
  const std::string& getName() const { return name_; }
  const std::string& getDescription() const { return description_; }

  // 获取参数模式
  virtual std::vector<SkillParam> getParams() const = 0;

 protected:
  std::string name_;
  std::string description_;
};

// 工具工厂
class NNDEPLOY_CC_API ToolFactory {
 public:
  static void registerTool(const std::string& name,
                          std::function<NodeTool*()> creator);
  static std::unique_ptr<NodeTool> createTool(const std::string& name);
  static std::vector<std::string> listTools();
};

}  // namespace dag
}  // namespace nndeploy
```

#### 3. Doc - 节点文档
```cpp
// framework/include/nndeploy/dag/node_doc.h

namespace nndeploy {
namespace dag {

// 节点文档
class NNDEPLOY_CC_API NodeDoc {
 public:
  NodeDoc(const std::string& node_type);

  // 设置基本信息
  void setTitle(const std::string& title);
  void setDescription(const std::string& description);
  void setCategory(const std::string& category);
  void setVersion(const std::string& version);
  void setAuthor(const std::string& author);

  // 添加输入/输出描述
  void addInput(const std::string& name, const std::string& type,
                const std::string& description);
  void addOutput(const std::string& name, const std::string& type,
                 const std::string& description);

  // 添加参数描述
  void addParam(const std::string& name, const std::string& type,
               const std::string& description,
               const std::string& default_value = "");

  // 添加示例
  void addExample(const std::string& title, const std::string& json_config);

  // 获取文档信息
  const std::string& getNodeType() const { return node_type_; }
  const std::string& getTitle() const { return title_; }
  const std::string& getDescription() const { return description_; }
  const std::string& getCategory() const { return category_; }

  // 生成 Markdown 文档
  std::string toMarkdown() const;

  // 生成 JSON 文档
  std::string toJSON() const;

 private:
  std::string node_type_;
  std::string title_;
  std::string description_;
  std::string category_;
  std::string version_;
  std::string author_;
  std::vector<std::tuple<std::string, std::string, std::string>> inputs_;
  std::vector<std::tuple<std::string, std::string, std::string>> outputs_;
  std::vector<std::tuple<std::string, std::string, std::string, std::string>> params_;
  std::vector<std::pair<std::string, std::string>> examples_;
};

}  // namespace dag
}  // namespace nndeploy
```

#### 4. Node 基类扩展
```cpp
// framework/include/nndeploy/dag/node.h (新增内容)

class NNDEPLOY_CC_API Node {
 public:
  // ... 现有接口 ...

  // Skill 相关接口
  virtual std::vector<NodeSkill> getSkills() const;
  virtual bool hasSkill(const std::string& skill_name) const;
  virtual base::Status executeSkill(const std::string& skill_name,
                                  const std::map<std::string, std::any>& params,
                                  std::any& result);

  // Tool 相关接口
  virtual std::vector<std::string> getTools() const;
  virtual ToolResult executeTool(const std::string& tool_name,
                               const std::map<std::string, std::any>& params);

  // Doc 相关接口
  virtual NodeDoc getDoc() const;
  virtual std::string getDocMarkdown() const;
  virtual std::string getDocJSON() const;

  // 注册文档（静态注册机制）
  static void registerDoc(const NodeDoc& doc);

 protected:
  std::vector<NodeSkill> skills_;
  std::map<std::string, std::unique_ptr<NodeTool>> tools_;
  NodeDoc doc_;
};
```

#### 5. 注册中心
```cpp
// framework/include/nndeploy/dag/registry.h

namespace nndeploy {
namespace dag {

// 节点能力注册中心
class NNDEPLOY_CC_API NodeRegistry {
 public:
  // 注册节点类型
  static void registerNode(const std::string& type, const std::string& description,
                        std::function<Node*()> creator);

  // 注册节点文档
  static void registerDoc(const NodeDoc& doc);

  // 查询节点
  static std::vector<std::string> listNodes();
  static NodeDoc getNodeDoc(const std::string& type);
  static std::vector<NodeSkill> getNodeSkills(const std::string& type);
  static std::vector<std::string> getNodeTools(const std::string& type);

  // 按类别查询
  static std::vector<std::string> listNodesByCategory(const std::string& category);

  // 搜索节点
  static std::vector<std::string> searchNodes(const std::string& keyword);

  // 导出所有节点文档
  static std::string exportAllDocsJSON();
  static std::string exportAllDocsMarkdown();

 private:
  static std::mutex registry_mutex_;
  static std::map<std::string, NodeDoc> docs_;
  static std::map<std::string, std::vector<NodeSkill>> skills_;
};

}  // namespace dag
}  // namespace nndeploy
```

### 核心操作流程

```
1. 节点注册阶段:
   a. 节点加载时调用 NodeRegistry::registerNode()
   b. 节点初始化时调用 registerDoc() 注册文档
   c. Tool 通过 ToolFactory 注册

2. 节点发现阶段:
   a. Agent 调用 NodeRegistry::searchNodes() 查找节点
   b. Agent 调用 getNodeDoc() 获取节点文档
   c. Agent 调用 getNodeSkills() 获取节点能力

3. 节点执行阶段:
   a. Agent 调用 executeSkill() 执行节点能力
   b. 节点可以调用 executeTool() 使用外部工具
```

### 技术细节
- 使用静态注册机制，节点加载时自动注册
- Skill 描述支持参数类型、默认值、验证函数
- Tool 支持异步执行（可选）
- Doc 支持导出为 Markdown 和 JSON 格式
- 提供搜索和过滤能力

## 5. 实施步骤

### Step 1: 基础数据结构定义
- 创建 `node_skill.h` 和 `node_skill.cc`
- 创建 `node_tool.h` 和 `node_tool.cc`
- 创建 `node_doc.h` 和 `node_doc.cc`
- 创建 `registry.h` 和 `registry.cc`
- 涉及文件: 新增文件

### Step 2: 扩展 Node 基类
- 修改 `framework/include/nndeploy/dag/node.h`
- 添加 Skill、Tool、Doc 相关接口
- 修改 `framework/source/nndeploy/dag/node.cc`
- 实现基本功能
- 涉及文件: `framework/include/nndeploy/dag/node.h`, `framework/source/nndeploy/dag/node.cc`

### Step 3: 实现 Tool 工厂和注册
- 实现 ToolFactory
- 添加工具注册机制
- 涉及文件: `framework/source/nndeploy/dag/node_tool.cc`

### Step 4: 实现 NodeRegistry
- 实现节点注册和查询
- 实现文档导出功能
- 添加搜索功能
- 涉及文件: `framework/source/nndeploy/dag/registry.cc`

### Step 5: 为现有插件添加示例
- 为 InferenceNode 添加文档
- 为 DetectNode 添加文档
- 为 SegmentNode 添加文档
- 涉及文件: 各插件目录

### Step 6: 添加单元测试
- 测试 Skill 注册和查询
- 测试 Tool 执行
- 测试 Doc 导出
- 涉及文件: `test/dag/`

### Step 7: 文档生成工具
- 创建命令行工具导出所有节点文档
- 生成 Markdown 格式的 API 文档
- 涉及文件: `tools/` (新增工具目录)

### 兼容性与迁移
- **向后兼容策略**: 新能力为可选特性，现有代码无需修改
- **迁移路径**: 逐步为现有节点添加文档和能力描述
- **过渡期安排**:
  - 阶段1: 基础设施完成
  - 阶段2: 为核心插件添加文档
  - 阶段3: 为所有插件添加文档

## 6. 验收标准

### 功能测试
- **测试用例 1**: Skill 注册和查询
  - 创建自定义节点并注册 Skill
  - 通过 Registry 查询 Skill
  - 验证 Skill 信息正确
- **测试用例 2**: Tool 执行
  - 注册一个 Tool
  - 执行 Tool 并验证结果
  - 验证参数验证
- **测试用例 3**: Doc 导出
  - 为节点添加完整文档
  - 导出为 Markdown
  - 导出为 JSON
- **测试用例 4**: 节点搜索
  - 按类别搜索节点
  - 按关键字搜索节点
  - 验证搜索结果正确
- **测试用例 5**: 现有插件兼容性
  - 运行现有 DAG 测试
  - 验证无破坏性变更

### 代码质量
- 代码通过编译
- 代码覆盖率 > 80%
- 符合现有代码规范
- API 文档完整

### 回归测试
- 现有 DAG 测试用例全部通过
- 现有插件测试全部通过
- 不影响工作流执行

### 性能与可维护性
- 注册开销 < 1ms/节点
- 查询开销 < 10ms
- 代码结构清晰

### 文档与示例
- API 文档完整
- 提供添加 Skill/Tool/Doc 的示例代码
- 提供命令行工具使用文档

## 7. 其他说明

### 相关资源
- 参考现有插件的配置文件结构
- 参考 JSON workflow 格式
- 参考现有的 NodeCreator 注册机制

### 风险与应对
- **潜在风险**: 元数据与实际实现不一致
  - 应对措施: 添加测试验证元数据与实际接口的一致性
- **潜在风险**: 注册机制可能影响启动时间
  - 应对措施: 使用惰性加载，需要时再加载详细信息
- **潜在风险**: 不同插件之间的 Skill 命名冲突
  - 应对措施: 使用命名空间或前缀区分

### 依赖关系
- **依赖**: 无
- **被依赖**: Agent 集成、工作流编排工具、自动文档生成

### 扩展性
- 支持动态加载/卸载节点
- 支持节点版本管理
- 支持节点依赖关系声明
- 支持 Skill 组合和链式调用

### 未来计划
- 支持 Skill 的输入/输出类型自动推导
- 支持基于 Skill 的自动工作流生成
- 支持 Tool 的远程调用（HTTP/RPC）
- 支持多语言文档（i18n）
