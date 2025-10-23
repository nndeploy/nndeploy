
# Production Environment Deployment

The nndeploy framework provides a complete solution from drag-and-drop development to production environment deployment, ensuring that algorithm models can run efficiently and stably in production environments.

- Development Phase: Design and debug workflows using the visual interface. **Develop custom nodes when necessary**
- Deployment Phase: Export JSON configuration and load via API

Whether through visual interface or API calls, all workflows execute in the unified high-performance C++ compute engine, ensuring consistency between development and deployment environments.

## Custom Input/Output Mode

**Use Cases**: Dynamic input data setting in code and custom output result processing

**Python LLM Example**:

```python
import nndeploy

# Create and load workflow
graph = nndeploy.dag.Graph("")
graph.remove_in_out_node()  # Remove input/output nodes for custom data flow
graph.load_file("path/to/llm_workflow.json")
graph.init()

# Set input data
input_edge = graph.get_input(0)
text = nndeploy.tokenizer.TokenizerText()
text.texts_ = ["<|im_start|>user\nPlease introduce NBA superstar Michael Jordan<|im_end|>\n<|im_start|>assistant\n"]
input_edge.set(text)

# Execute inference
status = graph.run()

# Get output results
output_edge = graph.get_output(0)
result = output_edge.get_graph_output()

# Clean up resources
graph.deinit()
```

**C++ LLM Example**:

```cpp
#include "nndeploy/dag/graph.h"

// Create and load workflow
std::shared_ptr<dag::Graph> graph = std::make_shared<dag::Graph>("");
base::Status status = graph->loadFile("path/to/llm_workflow.json");
graph->removeInOutNode();  // Remove input/output nodes
status = graph->init();

// Set input data
dag::Edge* input = graph->getInput(0);
tokenizer::TokenizerText* text = new tokenizer::TokenizerText();
text->texts_ = {
    "<|im_start|>user\nPlease introduce NBA superstar Michael Jordan<|im_end|>\n<|im_start|>assistant\n"
};
input->set(text, false);

// Execute inference
status = graph->run();

// Get output results
dag::Edge* output = graph->getOutput(0);
tokenizer::TokenizerText* result = output->getGraphOutput<tokenizer::TokenizerText>();

// Clean up resources
status = graph->deinit();
```

## Complete Workflow Mode

**Use Cases**: Workflow contains complete input/output processing logic, no additional data setup required

**Python Example**:

```python
import nndeploy

# Load and execute complete workflow
graph = nndeploy.dag.Graph("")
graph.load_file("path/to/llm_workflow.json")
graph.init()
status = graph.run()  # Workflow handles all input/output internally
graph.deinit()
```

**C++ Example**:

```cpp
// Load and execute complete workflow
std::shared_ptr<dag::Graph> graph = std::make_shared<dag::Graph>("");
base::Status status = graph->loadFile("path/to/llm_workflow.json");
status = graph->init();
status = graph->run();  // Workflow handles all input/output internally
status = graph->deinit();
```

---

## Example Code

| Algorithm Type | Python Example | C++ Example |
| -------------- | -------------- | ----------- |
| **Large Language Model** | [Python LLM](https://github.com/nndeploy/nndeploy/blob/main/demo/llm/demo.py) | [C++ LLM](https://github.com/nndeploy/nndeploy/blob/main/demo/llm/demo.cc) |
| **Object Detection** | [Python Detection](https://github.com/nndeploy/nndeploy/blob/main/demo/detect/demo.py) | [C++ Detection](https://github.com/nndeploy/nndeploy/blob/main/demo/detect/demo.cc) |


