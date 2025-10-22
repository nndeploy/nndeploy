
# nndeploy 使用指南

nndeploy 提供了灵活的工作流执行方式，支持从可视化设计到生产部署的完整流程。

## 🚀 快速开始

### 方式一：命令行执行

当你在可视化界面中完成工作流设计后，可以将其导出为 JSON 文件，然后通过命令行直接执行：

```bash
# Python 命令行工具
nndeploy-run-json --json_file path/to/workflow.json

# C++ 命令行工具
cd path/to/nndeploy
./build/nndeploy_demo_run_json --json_file path/to/workflow.json
```

**适用场景**：调试工作流、批处理任务、脚本自动化

---

### 方式二：API 编程调用

通过编程接口可以更灵活地控制工作流执行，支持自定义处理逻辑。

#### 🔧 自定义输入输出模式

**使用场景**：需要在代码中动态设置输入数据，或对输出结果进行自定义处理

**核心步骤**：
1. 移除工作流中的输入输出节点
2. 通过代码设置输入数据
3. 获取并处理输出结果

**Python 示例**：
```python
import nndeploy

# 创建并加载工作流
graph = nndeploy.dag.Graph("")
graph.remove_in_out_node()  # 移除输入输出节点，允许自定义数据流
graph.load_file("path/to/llm_workflow.json")
graph.init()

# 设置输入数据
input_edge = graph.get_input(0)    
text = nndeploy.tokenizer.TokenizerText()
text.texts_ = ["<|im_start|>user\nPlease introduce NBA superstar Michael Jordan<|im_end|>\n<|im_start|>assistant\n"]
input_edge.set(text)

# 执行推理
status = graph.run()

# 获取输出结果
output_edge = graph.get_output(0)
result = output_edge.get_graph_output()  

# 清理资源
graph.deinit()
```

**C++ 示例**：
```cpp
#include "nndeploy/dag/graph.h"

// 创建并加载工作流
std::shared_ptr<dag::Graph> graph = std::make_shared<dag::Graph>("");
base::Status status = graph->loadFile("path/to/llm_workflow.json");
graph->removeInOutNode();  // 移除输入输出节点
status = graph->init();

// 设置输入数据
dag::Edge* input = graph->getInput(0);
tokenizer::TokenizerText* text = new tokenizer::TokenizerText();
text->texts_ = {
    "<|im_start|>user\nPlease introduce NBA superstar Michael Jordan<|im_end|>\n<|im_start|>assistant\n"
};
input->set(text, false);

// 执行推理
status = graph->run();

// 获取输出结果
dag::Edge* output = graph->getOutput(0);
tokenizer::TokenizerText* result = output->getGraphOutput<tokenizer::TokenizerText>();

// 清理资源
status = graph->deinit();
```

#### 🎯 完整工作流模式

**使用场景**：工作流已包含完整的输入输出处理逻辑，无需额外的数据设置

**特点**：代码简洁，适合标准化的推理任务

**Python 示例**：
```python
import nndeploy

# 加载并执行完整工作流
graph = nndeploy.dag.Graph("")
graph.load_file("path/to/llm_workflow.json")
graph.init()
status = graph.run()  # 工作流内部处理所有输入输出
graph.deinit()
```

**C++ 示例**：
```cpp
// 加载并执行完整工作流
std::shared_ptr<dag::Graph> graph = std::make_shared<dag::Graph>("");
base::Status status = graph->loadFile("path/to/llm_workflow.json");
status = graph->init();
status = graph->run();  // 工作流内部处理所有输入输出
status = graph->deinit();
```

---

## 📚 更多示例

| 算法类型 | Python 示例 | C++ 示例 |
|---------|-------------|----------|
| **大语言模型** | [Python LLM](demo/llm/demo.py) | [C++ LLM](demo/llm/demo.cc) |
| **目标检测** | [Python 检测](demo/detect/demo.py) | [C++ 检测](demo/detect/demo.cc) |

---

## 💡 推荐开发流程

### 🔨 开发阶段
- **可视化设计**：使用 nndeploy 可视化界面拖拽构建工作流
- **实时调试**：在界面中实时验证算法效果和性能表现
- **节点扩展**：根据需要开发自定义算法节点

### 🚀 部署阶段  
- **一键导出**：将验证通过的工作流导出为 JSON 配置文件
- **生产集成**：通过 Python/C++ API 在生产环境中加载运行

无论通过可视化界面还是 API 调用，所有工作流都在底层统一的高性能 C++ 计算引擎中执行，**一致性保证**：开发调试和生产部署环境具有完全一致的执行行为和性能表现，实现 **"一次开发，处处运行"** 的理念，提升 AI 算法的落地效率