import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

@dataclass
class MockNode:
    name: str
    sleep_ms: int = 100       # 模拟计算耗时
    outputs: List[str] = field(default_factory=lambda: ["out"])

    async def run(self, data: Dict[str, Any]):
        await asyncio.sleep(self.sleep_ms / 1000)
        # 简单回传同样数据
        return {out: f"{self.name}_data" for out in self.outputs}


class MockGraph:
    """
    简易 DAG：按拓扑序依次执行节点（无并行 / 环）
    """

    def __init__(self, nodes: Dict[str, MockNode], edges: List[tuple]):
        self.nodes = nodes              # {id: MockNode}
        self.edges = edges              # [(src_id, dst_id, src_out, dst_in)]

        # 构建邻接表
        self.downstream: Dict[str, List[tuple]] = {nid: [] for nid in nodes}
        for src, dst, s_out, d_in in edges:
            self.downstream[src].append((dst, s_out, d_in))

        # 统计入度判拓扑
        indeg = {nid: 0 for nid in nodes}
        for _, dst, _, _ in edges:
            indeg[dst] += 1
        self._topsort = [nid for nid, d in indeg.items() if d == 0]  # 起始

        for nid in list(self._topsort):       # Kahn
            for dst, *_ in self.downstream[nid]:
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    self._topsort.append(dst)

        if len(self._topsort) != len(nodes):
            raise ValueError("MockGraph: detected cycle!")

    # ----------------------------------------
    async def run(
        self,
        callback: Callable[[str, float], None] | None = None,
    ):
        """
        顺序执行；每跑完一个节点执行 callback(node_id, progress)
        """
        data_cache: Dict[str, Dict[str, Any]] = {}
        total = len(self.nodes)

        for idx, nid in enumerate(self._topsort, 1):
            node = self.nodes[nid]
            in_data = data_cache.get(nid, {})
            out_data = await node.run(in_data)

            for dst, s_out, d_in in self.downstream[nid]:
                data_cache.setdefault(dst, {})[d_in] = out_data[s_out]

            if callback:
                callback(nid, idx / total)

        return data_cache
