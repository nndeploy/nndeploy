# llm

运行nndeploy前端拉出来的llm工作流

## run_json (完整的运行整个工作流)

```bash
cd path/to/nndeploy

# Python CLI
python3 demo/llm/demo.py --json_file resources/workflow/Detect_YOLO.json

# C++ CLI
./build/nndeploy_demo_detect --json_file resources/workflow/Detect_YOLO.json

# Result
查看工作流Encode节点对应的path_路径图片
```

## run_json_remove_in_out_node（开发者自己有输入输出逻辑，移除工作流中的输入和输出节点）

```bash

cd path/to/nndeploy

# Python CLI
python3 demo/llm/demo.py --json_file resources/workflow/Detect_YOLO.jsonn --remove_in_out_node --input_path path/to/input.jpg --output_path path/to/output.jpg

# C++ CLI
./build/nndeploy_demo_detect --json_file resources/workflow/Detect_YOLO.json --remove_in_out_node --input_path path/to/input.jpg --output_path path/to/output.jpg

# result
查看output_path_路径
```