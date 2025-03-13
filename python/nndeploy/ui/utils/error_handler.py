"""
错误处理工具模块

负责:
- 统一处理应用异常
- 提供友好错误提示
- 记录错误日志
- 支持错误恢复

错误提示采用模态对话框,包含详细信息
"""

import logging
import traceback
from typing import Optional, Callable, Any
import flet as ft

logger = logging.getLogger(__name__)

class ErrorLevel:
    """错误级别"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    FATAL = "fatal"

class ErrorHandler:
    """错误处理类"""
    
    def __init__(self):
        self._error_callbacks = []
        self._recovery_handlers = {}
        
    def register_callback(self, callback: Callable):
        """注册错误回调函数"""
        if callback not in self._error_callbacks:
            self._error_callbacks.append(callback)
            
    def register_recovery_handler(self, error_type: type, handler: Callable):
        """注册错误恢复处理器"""
        self._recovery_handlers[error_type] = handler
        
    def handle_error(
        self,
        error: Exception,
        level: str = ErrorLevel.ERROR,
        title: str = None,
        message: str = None,
        context: dict = None
    ):
        """处理错误
        
        Args:
            error: 异常对象
            level: 错误级别
            title: 错误标题
            message: 错误消息
            context: 错误上下文信息
        """
        # 获取错误信息
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "level": level,
            "title": title or type(error).__name__,
            "description": message or str(error),
            "context": context or {}
        }
        
        # 记录日志
        log_message = f"{error_info['title']}: {error_info['description']}"
        if level == ErrorLevel.FATAL:
            logger.critical(log_message, exc_info=error)
        elif level == ErrorLevel.ERROR:
            logger.error(log_message, exc_info=error)
        elif level == ErrorLevel.WARNING:
            logger.warning(log_message, exc_info=error)
        else:
            logger.info(log_message, exc_info=error)
            
        # 通知回调
        for callback in self._error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logger.error(f"错误回调执行失败: {e}")
                
        # 尝试恢复
        if type(error) in self._recovery_handlers:
            try:
                self._recovery_handlers[type(error)](error)
            except Exception as e:
                logger.error(f"错误恢复失败: {e}")
                
        return error_info
        
    def show_error_dialog(
        self,
        page: ft.Page,
        title: str,
        message: str,
        level: str = ErrorLevel.ERROR,
        details: str = None
    ):
        """显示错误对话框
        
        Args:
            page: Flet页面对象
            title: 对话框标题
            message: 错误消息
            level: 错误级别
            details: 详细信息
        """
        # 设置图标和颜色
        if level == ErrorLevel.ERROR:
            icon = ft.icons.ERROR
            color = "red"
        elif level == ErrorLevel.WARNING:
            icon = ft.icons.WARNING
            color = "orange"
        else:
            icon = ft.icons.INFO
            color = "blue"
            
        # 创建对话框
        dialog = ft.AlertDialog(
            title=ft.Text(title, size=20, weight=ft.FontWeight.BOLD),
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(icon, color=color, size=24),
                            ft.Text(message, size=16)
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    )
                ],
                tight=True,
            ),
            actions=[
                ft.TextButton("确定", on_click=lambda _: page.dialog.open = False),
            ],
        )
        
        # 如果有详细信息,添加展开面板
        if details:
            dialog.content.controls.append(
                ft.ExpansionTile(
                    title=ft.Text("详细信息"),
                    subtitle=ft.Text("点击展开查看详细错误信息"),
                    children=[
                        ft.Container(
                            ft.SelectableText(details),
                            padding=10,
                            bgcolor=ft.colors.BLACK12,
                            border_radius=5,
                        )
                    ],
                )
            )
            
        # 显示对话框
        page.dialog = dialog
        page.dialog.open = True
        page.update()

# 创建全局错误处理器实例
error_handler = ErrorHandler()

def handle_error(
    error: Exception,
    level: str = ErrorLevel.ERROR,
    title: str = None,
    message: str = None,
    context: dict = None
):
    """便捷的错误处理函数"""
    return error_handler.handle_error(error, level, title, message, context)

def show_error(
    page: ft.Page,
    title: str,
    message: str,
    level: str = ErrorLevel.ERROR,
    details: str = None
):
    """便捷的错误显示函数"""
    error_handler.show_error_dialog(page, title, message, level, details) 