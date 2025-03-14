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

# 获取模块级别的日志记录器
logger = logging.getLogger(__name__)

class ErrorLevel:
    """错误级别
    
    定义应用中使用的不同错误级别常量，用于区分错误的严重程度
    """
    INFO = "info"       # 信息级别，一般用于提示性消息
    WARNING = "warning" # 警告级别，表示潜在问题但不影响主要功能
    ERROR = "error"     # 错误级别，表示功能无法正常工作
    FATAL = "fatal"     # 致命错误，表示应用可能需要终止

class ErrorHandler:
    """错误处理类
    
    提供统一的错误处理机制，包括日志记录、错误提示和恢复机制
    """
    
    def __init__(self):
        """初始化错误处理器
        
        创建回调函数列表和错误恢复处理器映射
        """
        self._error_callbacks = []  # 存储错误发生时需要调用的回调函数
        self._recovery_handlers = {}  # 存储特定类型错误的恢复处理函数
        
    def register_callback(self, callback: Callable):
        """注册错误回调函数
        
        当错误发生时，会调用所有注册的回调函数
        
        Args:
            callback: 回调函数，接收错误信息字典作为参数
        """
        if callback not in self._error_callbacks:
            self._error_callbacks.append(callback)
            
    def register_recovery_handler(self, error_type: type, handler: Callable):
        """注册错误恢复处理器
        
        为特定类型的异常注册恢复处理函数
        
        Args:
            error_type: 异常类型
            handler: 处理函数，接收异常对象作为参数
        """
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
        
        完整处理异常，包括记录日志、通知回调和尝试恢复
        
        Args:
            error: 异常对象
            level: 错误级别，使用ErrorLevel中定义的常量
            title: 错误标题，如果为None则使用异常类名
            message: 错误消息，如果为None则使用异常的字符串表示
            context: 错误上下文信息，可包含任何相关的额外数据
            
        Returns:
            包含完整错误信息的字典
        """
        # 获取错误信息并构建错误信息字典
        error_info = {
            "type": type(error).__name__,  # 异常类型名称
            "message": str(error),         # 异常消息
            "traceback": traceback.format_exc(),  # 完整的堆栈跟踪
            "level": level,                # 错误级别
            "title": title or type(error).__name__,  # 使用提供的标题或默认为异常类名
            "description": message or str(error),    # 使用提供的消息或默认为异常消息
            "context": context or {}       # 错误上下文，默认为空字典
        }
        
        # 根据错误级别记录不同级别的日志
        log_message = f"{error_info['title']}: {error_info['description']}"
        if level == ErrorLevel.FATAL:
            logger.critical(log_message, exc_info=error)
        elif level == ErrorLevel.ERROR:
            logger.error(log_message, exc_info=error)
        elif level == ErrorLevel.WARNING:
            logger.warning(log_message, exc_info=error)
        else:  # INFO级别
            logger.info(log_message, exc_info=error)
            
        # 通知所有注册的回调函数
        for callback in self._error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logger.error(f"错误回调执行失败: {e}")
                
        # 尝试使用注册的恢复处理器恢复错误
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
        
        在UI中显示友好的错误提示对话框
        
        Args:
            page: Flet页面对象，用于显示对话框
            title: 对话框标题
            message: 错误消息，显示在对话框主体
            level: 错误级别，影响图标和颜色
            details: 详细信息，可选，如果提供则显示在可展开面板中
        """
        # 根据错误级别设置不同的图标和颜色
        if level == ErrorLevel.ERROR:
            icon = ft.Icons.ERROR
            color = "red"
        elif level == ErrorLevel.WARNING:
            icon = ft.Icons.WARNING
            color = "orange"
        else:  # INFO或其他级别
            icon = ft.Icons.INFO
            color = "blue"
            
        # 创建对话框组件
        dialog = ft.AlertDialog(
            title=ft.Text(title, size=20, weight=ft.FontWeight.BOLD),
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(icon, color=color, size=24),  # 错误图标
                            ft.Text(message, size=16)             # 错误消息
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    )
                ],
                tight=True,  # 紧凑布局
            ),
            actions=[
                ft.TextButton("确定", on_click=lambda _: self.close_dialog(page)),
            ],
        )
        
        # 如果提供了详细信息，添加可展开的详细信息面板
        if details:
            dialog.content.controls.append(
                ft.ExpansionTile(
                    title=ft.Text("详细信息"),
                    subtitle=ft.Text("点击展开查看详细错误信息"),
                    controls=[
                        ft.Container(
                            content=ft.Text(details, selectable=True),  # 使用Text的selectable属性替代SelectableText
                            padding=10,
                            bgcolor=ft.Colors.BLACK12,   # 浅灰色背景
                            border_radius=5,             # 圆角边框
                        )
                    ],
                )
            )
            
        # 在页面上显示对话框
        page.dialog = dialog
        page.dialog.open = True
        page.update()  # 更新UI

    def close_dialog(self, page):
        """关闭对话框
        
        关闭当前显示的对话框
        
        Args:
            page: Flet页面对象
        """
        page.dialog.open = False
        page.update()  # 更新UI以关闭对话框

# 创建全局错误处理器实例，应用中统一使用此实例
error_handler = ErrorHandler()

def handle_error(
    error: Exception,
    level: str = ErrorLevel.ERROR,
    title: str = None,
    message: str = None,
    context: dict = None
):
    """便捷的错误处理函数
    
    提供简化的接口调用全局错误处理器
    
    Args:
        error: 异常对象
        level: 错误级别
        title: 错误标题
        message: 错误消息
        context: 错误上下文信息
        
    Returns:
        包含完整错误信息的字典
    """
    return error_handler.handle_error(error, level, title, message, context)

def show_error(
    page: ft.Page,
    title: str,
    message: str,
    level: str = ErrorLevel.ERROR,
    details: str = None
):
    """便捷的错误显示函数
    
    提供简化的接口显示错误对话框
    
    Args:
        page: Flet页面对象
        title: 对话框标题
        message: 错误消息
        level: 错误级别
        details: 详细信息
    """
    error_handler.show_error_dialog(page, title, message, level, details) 
    
    
def demo_error_handling():
    """错误处理演示函数
    
    展示如何使用错误处理模块的各种功能
    """
    import flet as ft
    
    def main(page: ft.Page):
        page.title = "错误处理演示"
        page.theme_mode = ft.ThemeMode.LIGHT
        
        # 创建不同类型的错误演示按钮
        def show_info_error(e):
            show_error(
                page,
                "信息提示",
                "这是一个信息级别的提示",
                ErrorLevel.INFO,
                "这里可以放置更详细的信息说明"
            )
            
        def show_warning_error(e):
            show_error(
                page,
                "警告提示",
                "这是一个警告级别的提示",
                ErrorLevel.WARNING,
                "警告：操作可能导致数据丢失"
            )
            
        def show_error_dialog(e):
            show_error(
                page,
                "错误提示",
                "这是一个错误级别的提示",
                ErrorLevel.ERROR,
                "错误详情：无法连接到服务器"
            )
            
        def trigger_exception(e):
            try:
                # 故意触发异常
                result = 1 / 0
            except Exception as ex:
                error_info = handle_error(
                    ex,
                    ErrorLevel.ERROR,
                    "计算错误",
                    "执行除法运算时发生错误",
                    {"operation": "division", "value": 0}
                )
                show_error(
                    page,
                    error_info["title"],
                    error_info["message"],
                    error_info["level"],
                    error_info["traceback"]
                )
        
        # 创建演示界面
        page.add(
            ft.Text("错误处理演示", size=30, weight=ft.FontWeight.BOLD),
            ft.Text("点击下面的按钮查看不同类型的错误提示", size=16),
            ft.Divider(),
            ft.ElevatedButton("显示信息提示", on_click=show_info_error),
            ft.ElevatedButton("显示警告提示", on_click=show_warning_error),
            ft.ElevatedButton("显示错误提示", on_click=show_error_dialog),
            ft.ElevatedButton("触发异常并处理", on_click=trigger_exception),
        )
    
    ft.app(target=main, view=ft.WEB_BROWSER, port=9090)

if __name__ == "__main__":
    # 直接运行此文件可以启动演示
    demo_error_handling()
