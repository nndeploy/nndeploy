"""无数据边的 State 示例：Graph -> Loop -> Condition -> 分支 Graph。

安装本次构建的 nndeploy 后，在仓库根目录运行：
    python demo/dag/state_control.py
    python demo/dag/state_control.py --parallel task
"""

import argparse
from dataclasses import dataclass, field

from nndeploy.base import ParallelType, Status
from nndeploy.dag import Condition, Graph, Loop, Node, State


@dataclass
class DemoState(State):
    initial_value: int = 1
    iterations: int = 3
    step: int = 0
    value: int = 0
    history: list[int] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)


def require_ok(status, operation):
    if not status:
        raise RuntimeError(f"{operation} failed: {status}")


class Prepare(Node):
    def run(self):
        # 直接读写有类型声明的属性，不需要 key/value 接口。
        state: DemoState = self.state
        state.value = state.initial_value
        state.step = 0
        state.history.clear()
        state.trace.clear()
        state.trace.append("prepare")
        return Status.ok()


class Add(Node):
    def __init__(self, name, amount):
        super().__init__(name)
        self.amount = amount

    def run(self):
        self.state.value += self.amount
        self.state.trace.append(f"add({self.amount})")
        return Status.ok()


class Double(Node):
    def run(self):
        self.state.value *= 2
        self.state.trace.append("double")
        return Status.ok()


class Record(Node):
    def run(self):
        # list 原地修改也作用于同一份请求 State。
        self.state.history.append(self.state.value)
        self.state.trace.append(f"record({self.state.value})")
        return Status.ok()


class ChooseByStep(Condition):
    def choose(self):
        # 返回分支的注册索引：偶数轮选 0，奇数轮选 1。
        return self.state.step % 2


class Repeat(Loop):
    def loops(self):
        # 每轮重新读取上限；循环体和钩子也可以修改 iterations。
        return self.state.iterations

    def iter_before(self):
        self.state.trace.append(f"before({self.state.step})")
        return Status.ok()

    def iter_after(self):
        self.state.trace.append(f"after({self.state.step})")
        self.state.step += 1
        return Status.ok()


def build_graph(parallel_type=ParallelType.Sequential):
    # 分支 0：先加 2 再乘 2。故意反向注册，实际执行顺序由声明决定。
    even = Graph("even_branch")
    even.add_node(Double("double"))
    even.add_node(Add("add_two", 2))
    require_ok(even.set_execution_order(["add_two", "double"]), "order branch")

    choice = ChooseByStep("choose")
    choice.add_node(even)                     # 索引 0：分支 Graph
    choice.add_node(Add("add_ten", 10))       # 索引 1：单个节点
    # Condition 只选一个分支，不对候选分支设置执行顺序。

    repeat = Repeat("repeat")
    repeat.add_node(Record("record"))
    repeat.add_node(choice)
    require_ok(repeat.set_execution_order(["choose", "record"]), "order loop")

    graph = Graph("state_demo")
    graph.add_node(repeat)
    graph.add_node(Prepare("prepare"))
    require_ok(graph.set_execution_order(["prepare", "repeat"]), "order graph")
    # 父图的执行模式会传递给嵌套图；这里可选择串行或任务并行。
    graph.set_parallel_type(parallel_type)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", choices=("sequential", "task"), default="sequential")
    args = parser.parse_args()
    parallel_type = ParallelType.Task if args.parallel == "task" else ParallelType.Sequential
    graph = build_graph(parallel_type)
    state = DemoState(initial_value=1, iterations=3)

    try:
        require_ok(graph.init(), "init")
        # 只在根图绑定一次，所有嵌套节点通过 self.state 访问同一个对象。
        require_ok(graph.run_with_state(state), "run_with_state")
        assert state.history == [6, 16, 36], state.history
        assert state.value == 36 and state.step == 3
        print(f"value={state.value}, step={state.step}, history={state.history}")
        print("trace: " + " -> ".join(state.trace))
    finally:
        require_ok(graph.deinit(), "deinit")


if __name__ == "__main__":
    main()
