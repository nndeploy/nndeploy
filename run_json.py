"""nndeploy JSON 工作流单次执行入口。

功能：
- 从 JSON 文件加载并执行计算图，无需启动 HTTP 服务或编辑器。
- 支持节点参数覆盖、插件加载、并行执行方式、调试和图结构输出。
- 运行前需安装 nndeploy，并准备工作流所需的模型、插件及输入文件。

使用示例：
    python run_json.py --help
    python run_json.py --json_file ./graph.json
    python run_json.py --json_file ./graph.json --name my_graph
    python run_json.py --json_file ./graph.json --task_id task001
    python run_json.py --json_file ./graph.json --node_param node_name:param_key:param_value
    python run_json.py --json_file ./graph.json -np node_a:key_a:value_a node_b:key_b:value_b
    python run_json.py --json_file ./graph.json --parallel_type sequential
    python run_json.py --json_file ./graph.json --dump
    python run_json.py --json_file ./graph.json --debug
    python run_json.py --json_file ./graph.json --plugin ./plugins/custom.py ./plugins/custom.so
    python run_json.py --json_file ./graph.json --resources /path/to/resources
    python run_json.py --json-file ./graph.json --task-id task001 --parallel-type task

参数说明：
- --json_file / --json-file：必填，指定工作流 JSON 文件。
- --name：图名称；--task_id / --task-id：任务标识，二者默认均为空字符串。
- --node_param / --node-param / -np：可传多个 node:key:value 格式的节点参数。
  输入输出路径等参数使用具体节点定义的字段；包含空格时应为整个参数加引号。
- --parallel_type / --parallel-type：none、sequential、task、pipeline；也支持
  kParallelTypeNone、Sequential、Task、Pipeline。省略时保留工作流自身的设置。
- --dump 输出图结构；--debug 启用调试日志和图调试标记。
- --plugin：可指定多个插件路径，在执行前加载。
- --resources：将源目录复制到当前工作目录下的同名目录；若目标目录已存在，
  直接复用且不覆盖内容，也不会因此跳过工作流执行。此选项不会切换工作目录。
- 相对路径以当前工作目录为基准；工作流内的路径需与实际资源位置匹配。
- 执行成功退出码为 0；读取、插件加载或执行失败为 1；参数错误为 2。
"""

from nndeploy.dag.run_json import main


if __name__ == "__main__":
    raise SystemExit(main())
