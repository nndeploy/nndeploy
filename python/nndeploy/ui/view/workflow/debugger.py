"""
调试工具模块

负责:
- 提供工作流调试功能
- 显示调试信息和日志
- 支持断点设置
- 提供单步调试能力

日志区域支持语法高亮,错误信息用红色标识
"""

from typing import Dict, List, Optional, Callable
import flet as ft
from ...utils.logger import logger

class LogLevel:
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class LogEntry:
    """日志条目"""
    
    def __init__(
        self,
        level: str,
        message: str,
        node_id: Optional[str] = None,
        timestamp: Optional[str] = None
    ):
        self.level = level
        self.message = message
        self.node_id = node_id
        self.timestamp = timestamp
        
    @property
    def color(self) -> str:
        """获取日志颜色"""
        colors = {
            LogLevel.DEBUG: ft.colors.BLUE_400,
            LogLevel.INFO: ft.colors.GREEN_400,
            LogLevel.WARNING: ft.colors.ORANGE_400,
            LogLevel.ERROR: ft.colors.RED_400
        }
        return colors.get(self.level, ft.colors.BLACK54)

class Debugger(ft.UserControl):
    """调试器"""
    
    def __init__(
        self,
        on_step: Optional[Callable[[], None]] = None,
        on_continue: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None
    ):
        super().__init__()
        self.on_step = on_step
        self.on_continue = on_continue
        self.on_stop = on_stop
        
        # 调试状态
        self._paused = False
        self._breakpoints: Dict[str, bool] = {}
        self._logs: List[LogEntry] = []
        self._auto_scroll = True
        
    def build(self):
        return ft.Column(
            [
                # 工具栏
                ft.Row(
                    [
                        ft.IconButton(
                            icon=ft.icons.PLAY_ARROW,
                            tooltip="继续执行",
                            disabled=not self._paused,
                            on_click=lambda _: (
                                self.on_continue and self.on_continue()
                            )
                        ),
                        ft.IconButton(
                            icon=ft.icons.SKIP_NEXT,
                            tooltip="单步执行",
                            disabled=not self._paused,
                            on_click=lambda _: (
                                self.on_step and self.on_step()
                            )
                        ),
                        ft.IconButton(
                            icon=ft.icons.STOP,
                            tooltip="停止执行",
                            on_click=lambda _: (
                                self.on_stop and self.on_stop()
                            )
                        ),
                        ft.VerticalDivider(),
                        ft.IconButton(
                            icon=ft.icons.DELETE_SWEEP,
                            tooltip="清空日志",
                            on_click=self._clear_logs
                        ),
                        ft.IconButton(
                            icon=ft.icons.VERTICAL_ALIGN_BOTTOM,
                            tooltip="自动滚动",
                            selected=self._auto_scroll,
                            on_click=self._toggle_auto_scroll
                        )
                    ],
                    alignment=ft.MainAxisAlignment.START
                ),
                
                # 日志区域
                ft.Container(
                    content=ft.ListView(
                        [self._build_log_entry(log) for log in self._logs],
                        spacing=2,
                        auto_scroll=self._auto_scroll
                    ),
                    bgcolor=ft.colors.BLACK,
                    border_radius=5,
                    padding=10,
                    expand=True
                )
            ],
            spacing=10,
            expand=True
        )
        
    def _build_log_entry(self, log: LogEntry) -> ft.Control:
        """构建日志条目"""
        return ft.SelectableText(
            f"[{log.timestamp}] {log.message}",
            color=log.color,
            size=12,
            font_family="Consolas"
        )
        
    def _clear_logs(self, _):
        """清空日志"""
        self._logs.clear()
        self.update()
        
    def _toggle_auto_scroll(self, _):
        """切换自动滚动"""
        self._auto_scroll = not self._auto_scroll
        self.update()
        
    def add_log(
        self,
        level: str,
        message: str,
        node_id: Optional[str] = None
    ):
        """添加日志"""
        from datetime import datetime
        
        log = LogEntry(
            level=level,
            message=message,
            node_id=node_id,
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3]
        )
        self._logs.append(log)
        self.update()
        
    def set_paused(self, paused: bool):
        """设置暂停状态"""
        self._paused = paused
        self.update()
        
    def toggle_breakpoint(self, node_id: str) -> bool:
        """切换断点状态"""
        self._breakpoints[node_id] = not self._breakpoints.get(node_id, False)
        return self._breakpoints[node_id]
        
    def has_breakpoint(self, node_id: str) -> bool:
        """检查是否有断点"""
        return self._breakpoints.get(node_id, False) 