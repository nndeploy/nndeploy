"""
性能监控模块

负责:
- 监控系统资源使用
- 统计节点执行时间
- 提供性能数据图表
- 支持性能优化建议

图表采用简洁设计,关键指标用不同颜色区分
"""

from typing import Dict, List, Optional
import flet as ft
from ...utils.performance_monitor import PerformanceMetrics

class PerformanceChart(ft.UserControl):
    """性能图表"""
    
    def __init__(self, title: str, max_points: int = 100):
        super().__init__()
        self.title = title
        self.max_points = max_points
        self._points: List[float] = []
        
    def build(self):
        return ft.Column(
            [
                # 标题
                ft.Text(
                    self.title,
                    size=14,
                    weight=ft.FontWeight.BOLD
                ),
                
                # 图表
                ft.Container(
                    content=ft.LineChart(
                        data_series=[
                            ft.LineChartData(
                                self._points,
                                color=ft.colors.BLUE,
                                stroke_width=2
                            )
                        ],
                        border=ft.border.all(1, ft.colors.BLACK26),
                        horizontal_grid_lines=True,
                        vertical_grid_lines=True,
                        tooltip_bgcolor=ft.colors.with_opacity(0.8, ft.colors.BLACK)
                    ),
                    height=100
                )
            ],
            spacing=10
        )
        
    def add_point(self, value: float):
        """添加数据点"""
        self._points.append(value)
        if len(self._points) > self.max_points:
            self._points.pop(0)
        self.update()

class NodePerformance(ft.UserControl):
    """节点性能"""
    
    def __init__(self, node_id: str, node_name: str):
        super().__init__()
        self.node_id = node_id
        self.node_name = node_name
        self._execution_time = 0.0
        self._memory_usage = 0
        
    def build(self):
        return ft.ListTile(
            leading=ft.Icon(ft.icons.TIMER),
            title=ft.Text(self.node_name),
            subtitle=ft.Text(
                f"执行时间: {self._execution_time:.2f}ms  "
                f"内存使用: {self._memory_usage / 1024 / 1024:.1f}MB"
            )
        )
        
    def update_metrics(self, execution_time: float, memory_usage: int):
        """更新性能指标"""
        self._execution_time = execution_time
        self._memory_usage = memory_usage
        self.update()

class PerformanceMonitor(ft.UserControl):
    """性能监控器"""
    
    def __init__(self):
        super().__init__()
        self._cpu_chart = PerformanceChart("CPU使用率")
        self._memory_chart = PerformanceChart("内存使用率")
        self._node_metrics: Dict[str, NodePerformance] = {}
        
    def build(self):
        return ft.Column(
            [
                # 系统资源监控
                ft.Row(
                    [
                        ft.Container(
                            content=self._cpu_chart,
                            expand=True
                        ),
                        ft.Container(
                            content=self._memory_chart,
                            expand=True
                        )
                    ],
                    spacing=20
                ),
                
                ft.Divider(),
                
                # 节点性能列表
                ft.Column(
                    [
                        node_perf
                        for node_perf in self._node_metrics.values()
                    ],
                    scroll=ft.ScrollMode.AUTO,
                    spacing=5
                )
            ],
            spacing=20
        )
        
    def update_metrics(self, metrics: PerformanceMetrics):
        """更新性能指标"""
        # 更新系统资源图表
        self._cpu_chart.add_point(metrics.cpu_percent)
        self._memory_chart.add_point(
            metrics.memory_used / metrics.memory_total * 100
        )
        self.update()
        
    def update_node_metrics(
        self,
        node_id: str,
        node_name: str,
        execution_time: float,
        memory_usage: int
    ):
        """更新节点性能指标"""
        if node_id not in self._node_metrics:
            self._node_metrics[node_id] = NodePerformance(
                node_id,
                node_name
            )
            
        self._node_metrics[node_id].update_metrics(
            execution_time,
            memory_usage
        )
        self.update()
        
    def clear_node_metrics(self):
        """清空节点性能指标"""
        self._node_metrics.clear()
        self.update() 