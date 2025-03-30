"""
主菜单模块

负责:
- 整合所有菜单组件
- 提供统一的菜单栏界面
- 管理菜单项之间的交互
- 处理菜单事件的分发

菜单栏采用水平布局,左对齐,确保在不同屏幕尺寸下保持一致的视觉效果
"""

import flet as ft
from typing import Optional, Callable, Dict, Any

from file_menu import FileMenu
from edit_menu import EditMenu
from settings_menu import SettingsMenu
from help_menu import HelpMenu
from nndeploy.ui.config.language import get_text
from nndeploy.ui.config.theme import get_color

# 用submenubar or popumenubar

class MenuState:
    """菜单状态管理类"""
    def __init__(self):
        self.can_undo = False
        self.can_redo = False
        self.has_selection = False
        self.has_clipboard = False
        self.recent_files = []

def Menu(
    on_file_new: Optional[Callable[[], None]] = None,
    on_file_open: Optional[Callable[[], None]] = None,
    on_file_save: Optional[Callable[[], None]] = None,
    on_file_save_as: Optional[Callable[[], None]] = None,
    on_file_import: Optional[Callable[[], None]] = None,
    on_file_export: Optional[Callable[[], None]] = None,
    on_edit_undo: Optional[Callable[[], None]] = None,
    on_edit_redo: Optional[Callable[[], None]] = None,
    on_edit_cut: Optional[Callable[[], None]] = None,
    on_edit_copy: Optional[Callable[[], None]] = None,
    on_edit_paste: Optional[Callable[[], None]] = None,
    on_edit_delete: Optional[Callable[[], None]] = None,
    on_edit_select_all: Optional[Callable[[], None]] = None,
    on_settings_canvas: Optional[Callable[[], None]] = None,
    on_settings_node: Optional[Callable[[], None]] = None,
    on_settings_edge: Optional[Callable[[], None]] = None,
    on_help_documentation: Optional[Callable[[], None]] = None,
    on_help_tutorial: Optional[Callable[[], None]] = None,
    on_help_feedback: Optional[Callable[[], None]] = None,
    on_help_about: Optional[Callable[[], None]] = None,
    menu_state: Optional[MenuState] = None
):
    """
    创建主菜单栏
    
    参数:
        各种菜单项的回调函数
        menu_state: 菜单状态对象
    
    返回:
        包含所有菜单项的菜单栏控件
    """
    if menu_state is None:
        menu_state = MenuState()
    
    file_menu = FileMenu(
        on_new=on_file_new,
        on_open=on_file_open,
        on_save=on_file_save,
        on_save_as=on_file_save_as,
        on_import=on_file_import,
        on_export=on_file_export,
        recent_files=menu_state.recent_files
    )
    
    edit_menu = EditMenu(
        on_undo=on_edit_undo,
        on_redo=on_edit_redo,
        on_cut=on_edit_cut,
        on_copy=on_edit_copy,
        on_paste=on_edit_paste,
        on_delete=on_edit_delete,
        on_select_all=on_edit_select_all,
        can_undo=menu_state.can_undo,
        can_redo=menu_state.can_redo,
        has_selection=menu_state.has_selection,
        has_clipboard=menu_state.has_clipboard
    )
    
    settings_menu = SettingsMenu(
        on_canvas_settings=on_settings_canvas,
        on_node_settings=on_settings_node,
        on_edge_settings=on_settings_edge
    )
    
    help_menu = HelpMenu(
        on_documentation=on_help_documentation,
        on_tutorial=on_help_tutorial,
        on_feedback=on_help_feedback,
        on_about=on_help_about
    )
    
    return ft.Container(
        content=ft.Row(
            controls=[
                file_menu,
                edit_menu,
                settings_menu,
                help_menu,
            ],
            alignment=ft.MainAxisAlignment.START,
            spacing=2,
        ),
        padding=ft.padding.only(left=8, right=8),
        bgcolor=get_color("menu.background"),
        border_radius=ft.border_radius.only(
            bottom_left=4,
            bottom_right=4
        ),
        shadow=ft.BoxShadow(
            spread_radius=1,
            blur_radius=3,
            color=ft.colors.with_opacity(0.2, "black"),
            offset=ft.Offset(0, 1)
        )
    )

# 预览演示
def MenuDemo(page: ft.Page):
    """菜单预览演示"""
    
    # 状态显示区
    status_text = ft.Text("菜单操作状态将显示在这里")
    
    def update_status(action: str):
        status_text.value = f"执行操作: {action}"
        page.update()
    
    # 创建菜单状态
    menu_state = MenuState()
    menu_state.can_undo = True
    menu_state.can_redo = True
    menu_state.has_selection = True
    menu_state.has_clipboard = True
    menu_state.recent_files = ["项目1.nnd", "示例2.nnd", "测试3.nnd"]
    
    # 创建菜单
    menu = Menu(
        on_file_new=lambda: update_status("新建文件"),
        on_file_open=lambda: update_status("打开文件"),
        on_file_save=lambda: update_status("保存文件"),
        on_file_save_as=lambda: update_status("另存为"),
        on_file_import=lambda: update_status("导入"),
        on_file_export=lambda: update_status("导出"),
        on_edit_undo=lambda: update_status("撤销"),
        on_edit_redo=lambda: update_status("重做"),
        on_edit_cut=lambda: update_status("剪切"),
        on_edit_copy=lambda: update_status("复制"),
        on_edit_paste=lambda: update_status("粘贴"),
        on_edit_delete=lambda: update_status("删除"),
        on_edit_select_all=lambda: update_status("全选"),
        on_settings_canvas=lambda: update_status("画布设置"),
        on_settings_node=lambda: update_status("节点设置"),
        on_settings_edge=lambda: update_status("边缘设置"),
        on_help_documentation=lambda: update_status("查看文档"),
        on_help_tutorial=lambda: update_status("查看教程"),
        on_help_feedback=lambda: update_status("提交反馈"),
        on_help_about=lambda: update_status("关于"),
        menu_state=menu_state
    )
    
    # 创建演示内容
    content = ft.Container(
        content=ft.Column([
            ft.Text("NNDeploy 菜单演示", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("点击上方菜单项查看效果", italic=True),
            ft.Divider(),
            status_text,
        ]),
        padding=20,
        alignment=ft.alignment.center,
        expand=True
    )
    
    # 设置页面
    page.title = "NNDeploy 菜单演示"
    page.add(
        ft.Column([
            menu,
            content
        ], spacing=0, expand=True)
    )

if __name__ == "__main__":
    ft.app(target=MenuDemo, view=ft.WEB_BROWSER, port=9090)
