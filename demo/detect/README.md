# llm

运行nndeploy前端拉出来的检测工作流

## run_json (完整的运行整个工作流)

```bash
cd path/to/nndeploy

# Python CLI，保证当前工作目录下有resource资源
python3 demo/llm/demo.py --json_file resources/workflow/Detect_YOLO.json
# Python CLI，会将path/to/resources拷贝到当前工作目录
python3 demo.py --json_file path/to/resources/workflow/Detect_YOLO.json --resources path/to/resources

# C++ CLI，保证当前工作目录下有resource资源
./build/nndeploy_demo_detect --json_file resources/workflow/Detect_YOLO.json
# C++ CLI，会将path/to/resources拷贝到当前工作目录
./nndeploy_demo_detect --json_file path/to/resources/workflow/Detect_YOLO.json --resources path/to/resources

# Result
查看工作流Encode节点对应的path_路径图片
```

## run_json_remove_in_out_node（开发者自己有输入输出逻辑，移除工作流中的输入和输出节点）

```bash

cd path/to/nndeploy

# Python CLI，保证当前工作目录下有resource资源
python3 demo/llm/demo.py --json_file resources/workflow/Detect_YOLO.jsonn --remove_in_out_node --input_path path/to/input.jpg --output_path path/to/output.jpg
# Python CLI，会将path/to/resources拷贝到当前工作目录
python3 demo.py --json_file path/to/resources/workflow/Detect_YOLO.jsonn --remove_in_out_node --input_path path/to/input.jpg --output_path path/to/output.jpg --resources path/to/resources

# C++ CLI，保证当前工作目录下有resource资源
./build/nndeploy_demo_detect --json_file resources/workflow/Detect_YOLO.json --remove_in_out_node --input_path path/to/input.jpg --output_path path/to/output.jpg
# C++ CLI，会将path/to/resources拷贝到当前工作目录
./build/nndeploy_demo_detect --json_file resources/workflow/Detect_YOLO.json --remove_in_out_node --input_path path/to/input.jpg --output_path path/to/output.jpg --resources path/to/resources

# result
查看output_path_路径
```