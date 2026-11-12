"""混用 Edge 数据流、请求 State 和手动执行顺序，无需模型或 GPU。

安装本次构建的 nndeploy 后，在仓库根目录运行：
    python demo/dag/state_edge.py
    python demo/dag/state_edge.py --parallel task
"""

import argparse
from dataclasses import dataclass, field

from nndeploy.base import ParallelType, Status
from nndeploy.dag import Edge, Graph, Node, State


@dataclass
class RequestState(State):
    bias: int = 10
    offset: int = 0
    prepared: bool = False
    total: int = 0
    count: int = 0
    summary: str = ""
    trace: list[str] = field(default_factory=list)


def require_ok(status, operation):
    if not status:
        raise RuntimeError(f"{operation} failed: {status}")


class LegacyDouble(Node):
    """沿用原有数据流接口：只读写 Edge，无需知道 State 的存在。"""

    def run(self):
        values = self.get_input_data()
        return self.set_output_data([value * 2 for value in values])


class PrepareState(Node):
    """无输入输出边，通过显式顺序加入执行。"""

    def run(self):
        self.state.offset = self.state.bias
        self.state.prepared = True
        self.state.trace.append("prepare")
        return Status.ok()


class ApplyOffset(Node):
    """Edge 传递数据，State 提供请求配置并记录统计结果。"""

    def run(self):
        state: RequestState = self.state
        assert state.prepared, "prepare must finish before apply_offset"
        values = self.get_input_data()
        result = [value + state.offset for value in values]
        require_ok(self.set_output_data(result), "publish result")
        state.total = sum(result)
        state.count = len(result)
        state.trace.append("apply_offset")
        return Status.ok()


class SummarizeState(Node):
    """无数据边，等待 ApplyOffset 完成后读取它写入的 State。"""

    def run(self):
        self.state.summary = f"count={self.state.count}, total={self.state.total}"
        self.state.trace.append("summarize")
        return Status.ok()


def build_graph(parallel_type=ParallelType.Sequential):
    source = Edge("input")
    doubled = Edge("doubled")
    result = Edge("result")
    graph = Graph("state_edge_demo", [source], [result])
    # 构造函数中的原生 Edge 指针不持有 Python 对象，保留引用至图释放。
    graph._edge_refs = (source, doubled, result)

    # 故意反向注册节点；执行时由数据依赖和控制依赖共同决定顺序。
    graph.add_node(SummarizeState("summarize"))
    graph.add_node(ApplyOffset("apply_offset", [doubled], [result]))
    graph.add_node(LegacyDouble("double", [source], [doubled]))
    graph.add_node(PrepareState("prepare"))

    # double 不在列表里，仍因 Edge 连接参与执行。
    # apply_offset 等待 double（数据依赖）和 prepare（控制依赖）都完成。
    require_ok(
        graph.set_execution_order(["prepare", "apply_offset", "summarize"]),
        "set_execution_order",
    )
    graph.set_parallel_type(parallel_type)
    return graph, source, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", choices=("sequential", "task"), default="sequential")
    args = parser.parse_args()
    parallel_type = ParallelType.Task if args.parallel == "task" else ParallelType.Sequential
    graph, source, result = build_graph(parallel_type)

    # 同一个已初始化的图连续处理两次请求，每次替换输入并创建独立 State。
    requests = [([1, 2, 3], 10, [12, 14, 16]), ([4, 5], -1, [7, 9])]
    states = []
    outputs = []
    try:
        require_ok(graph.init(), "init")
        for index, (values, bias, expected) in enumerate(requests, start=1):
            state = RequestState(bias=bias)
            require_ok(source.set(values), "set input Edge")
            require_ok(graph.run_with_state(state), "run_with_state")
            output = result.get_graph_output()

            # 同时验证数据边结果、共享状态以及手动指定的执行顺序。
            assert output == expected, output
            assert state.offset == bias and state.prepared
            assert state.total == sum(expected) and state.count == len(expected)
            assert state.summary == f"count={len(expected)}, total={sum(expected)}"
            assert state.trace == ["prepare", "apply_offset", "summarize"]
            states.append(state)
            outputs.append(output)
            print(f"request {index}: output={output}, {state.summary}")
            print("trace: " + " -> ".join(state.trace))

        # 第二次运行不能修改第一次请求的结果或默认列表。
        assert outputs[0] == [12, 14, 16]
        assert states[0].total == 42 and states[0].offset == 10
        assert states[0].trace == ["prepare", "apply_offset", "summarize"]
        assert states[0].trace is not states[1].trace
    finally:
        require_ok(graph.deinit(), "deinit")


if __name__ == "__main__":
    main()
