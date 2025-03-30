"""
执行控制模块

负责:
- 控制工作流的执行
- 管理执行状态
- 提供执行进度显示
- 处理执行异常

执行按钮采用dify的蓝色,状态用不同颜色标识
"""

from typing import Dict, Optional, Callable
import flet as ft
from ...config.language import get_text
from ...service.execution_service import ExecutionStatus

class ExecutionProgress(ft.UserControl):
    """执行进度"""
    
    def __init__(self):
        super().__init__()
        self._total = 0
        self._completed = 0
        self._status = ExecutionStatus.READY
        
    def build(self):
        return ft.Column(
            [
                # 状态指示器
                ft.Container(
                    content=ft.Row(
                        [
                            ft.ProgressRing(
                                width=16,
                                height=16,
                                stroke_width=2,
                                color=self._get_status_color()
                            ) if self._status == ExecutionStatus.RUNNING else None,
                            ft.Icon(
                                self._get_status_icon(),
                                color=self._get_status_color(),
                                size=16
                            ),
                            ft.Text(
                                self._get_status_text(),
                                color=self._get_status_color(),
                                size=12
                            )
                        ],
                        spacing=5,
                        alignment=ft.MainAxisAlignment.START
                    ),
                    padding=5
                ),
                
                # 进度条
                ft.ProgressBar(
                    width=200,
                    value=self._completed / self._total if self._total > 0 else 0,
                    color=self._get_status_color(),
                    bgcolor=ft.colors.with_opacity(0.2, self._get_status_color())
                ) if self._total > 0 else None
            ],
            spacing=5
        )
        
    def _get_status_icon(self) -> str:
        """获取状态图标"""
        icons = {
            ExecutionStatus.READY: ft.icons.PLAY_CIRCLE_OUTLINE,
            ExecutionStatus.RUNNING: ft.icons.REFRESH,
            ExecutionStatus.PAUSED: ft.icons.PAUSE_CIRCLE_OUTLINE,
            ExecutionStatus.COMPLETED: ft.icons.CHECK_CIRCLE_OUTLINE,
            ExecutionStatus.FAILED: ft.icons.ERROR_OUTLINE,
            ExecutionStatus.CANCELED: ft.icons.CANCEL
        }
        return icons.get(self._status, ft.icons.HELP_OUTLINE)
        
    def _get_status_color(self) -> str:
        """获取状态颜色"""
        colors = {
            ExecutionStatus.READY: ft.colors.BLUE,
            ExecutionStatus.RUNNING: ft.colors.BLUE,
            ExecutionStatus.PAUSED: ft.colors.ORANGE,
            ExecutionStatus.COMPLETED: ft.colors.GREEN,
            ExecutionStatus.FAILED: ft.colors.RED,
            ExecutionStatus.CANCELED: ft.colors.GREY
        }
        return colors.get(self._status, ft.colors.GREY)
        
    def _get_status_text(self) -> str:
        """获取状态文本"""
        return get_text(f"execution.status.{self._status.value}")
        
    def update_progress(self, completed: int, total: int):
        """更新进度"""
        self._completed = completed
        self._total = total
        self.update()
        
    def update_status(self, status: ExecutionStatus):
        """更新状态"""
        self._status = status
        self.update()

class Executor(ft.UserControl):
    """执行控制器"""
    
    def __init__(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_pause: Optional[Callable[[], None]] = None,
        on_resume: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None
    ):
        super().__init__()
        self.on_start = on_start
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.on_stop = on_stop
        
        self._status = ExecutionStatus.READY
        self._progress = ExecutionProgress()
        
    def build(self):
        return ft.Row(
            [
                # 控制按钮
                ft.IconButton(
                    icon=ft.icons.PLAY_ARROW if self._status == ExecutionStatus.READY
                    else ft.icons.PAUSE if self._status == ExecutionStatus.RUNNING
                    else ft.icons.PLAY_ARROW if self._status == ExecutionStatus.PAUSED
                    else ft.icons.REFRESH,
                    tooltip=get_text(
                        "execution.start" if self._status == ExecutionStatus.READY
                        else "execution.pause" if self._status == ExecutionStatus.RUNNING
                        else "execution.resume" if self._status == ExecutionStatus.PAUSED
                        else "execution.restart"
                    ),
                    on_click=self._on_primary_action
                ),
                ft.IconButton(
                    icon=ft.icons.STOP,
                    tooltip=get_text("execution.stop"),
                    disabled=self._status not in [
                        ExecutionStatus.RUNNING,
                        ExecutionStatus.PAUSED
                    ],
                    on_click=lambda _: self.on_stop and self.on_stop()
                ),
                
                ft.VerticalDivider(),
                
                # 进度指示器
                self._progress
            ],
            spacing=0
        )
        
    def _on_primary_action(self, _):
        """主要动作按钮点击"""
        if self._status == ExecutionStatus.READY:
            # 开始执行
            if self.on_start:
                self.on_start()
                
        elif self._status == ExecutionStatus.RUNNING:
            # 暂停执行
            if self.on_pause:
                self.on_pause()
                
        elif self._status == ExecutionStatus.PAUSED:
            # 恢复执行
            if self.on_resume:
                self.on_resume()
                
        else:
            # 重新开始
            if self.on_start:
                self.on_start()
                
    def update_status(self, status: ExecutionStatus):
        """更新状态"""
        self._status = status
        self._progress.update_status(status)
        self.update()
        
    def update_progress(self, completed: int, total: int):
        """更新进度"""
        self._progress.update_progress(completed, total) 