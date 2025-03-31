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
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from file_menu import FileMenu, update_recent_files, register_recent_files_callback
from edit_menu import EditMenu
from settings_menu import SettingsMenu
from help_menu import HelpMenu
from nndeploy.ui.config.language import get_text, language_config
from nndeploy.ui.config.theme import get_color
from nndeploy.ui.config.shortcuts import get_shortcut

class MenuState:
    """菜单状态管理类"""
    def __init__(self):
        self.can_undo: bool = False
        self.can_redo: bool = False
        self.has_selection: bool = False
        self.has_clipboard: bool = False
        self.recent_files: List[str] = []
        self.is_dirty: bool = False  # 文件是否被修改
        
    def update(self, **kwargs):
        """更新菜单状态"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

class Menu:
    """主菜单类"""
    def __init__(
        self,
        page: ft.Page,
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
        on_language_settings: Optional[Callable[[], None]] = None,
        on_theme_settings: Optional[Callable[[], None]] = None,
        on_auto_save_settings: Optional[Callable[[], None]] = None,
        on_help_documentation: Optional[Callable[[], None]] = None,
        on_help_feedback: Optional[Callable[[], None]] = None,
        on_help_about: Optional[Callable[[], None]] = None,
    ):
        self.page = page
        self.menu_state = MenuState()
        self.callbacks = {
            'file_new': on_file_new,
            'file_open': on_file_open,
            'file_save': on_file_save,
            'file_save_as': on_file_save_as,
            'file_import': on_file_import,
            'file_export': on_file_export,
            'edit_undo': on_edit_undo,
            'edit_redo': on_edit_redo,
            'edit_cut': on_edit_cut,
            'edit_copy': on_edit_copy,
            'edit_paste': on_edit_paste,
            'edit_delete': on_edit_delete,
            'edit_select_all': on_edit_select_all,
            'settings_canvas': on_language_settings,
            'settings_node': on_theme_settings,
            'settings_edge': on_auto_save_settings,
            'help_documentation': on_help_documentation,
            'help_feedback': on_help_feedback,
            'help_about': on_help_about,
        }
        
        # 注册最近文件更新回调
        register_recent_files_callback(self._on_recent_files_update)
        
        # 注册快捷键
        self._register_shortcuts()
        
        # 创建UI组件
        self.container = self._build_ui()
        
    def _build_ui(self) -> ft.Container:
        """构建菜单UI"""
        file_menu = FileMenu(
            on_new=self.callbacks['file_new'],
            on_open=self.callbacks['file_open'],
            on_save=self.callbacks['file_save'],
            on_save_as=self.callbacks['file_save_as'],
            on_import=self.callbacks['file_import'],
            on_export=self.callbacks['file_export'],
            recent_files=self.menu_state.recent_files
        )
        
        edit_menu = EditMenu(
            on_undo=self.callbacks['edit_undo'],
            on_redo=self.callbacks['edit_redo'],
            on_cut=self.callbacks['edit_cut'],
            on_copy=self.callbacks['edit_copy'],
            on_paste=self.callbacks['edit_paste'],
            on_delete=self.callbacks['edit_delete'],
            on_select_all=self.callbacks['edit_select_all'],
            can_undo=self.menu_state.can_undo,
            can_redo=self.menu_state.can_redo,
            has_selection=self.menu_state.has_selection,
            has_clipboard=self.menu_state.has_clipboard
        )
        
        settings_menu = SettingsMenu(
            on_language_settings=self.callbacks['settings_canvas'],
            on_theme_settings=self.callbacks['settings_node'],
            on_auto_save_settings=self.callbacks['settings_edge']
        )
        
        help_menu = HelpMenu(
            on_documentation=self.callbacks['help_documentation'],
            on_feedback=self.callbacks['help_feedback'],
            on_about=self.callbacks['help_about']
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
        
    def _register_shortcuts(self):
        """注册快捷键"""
        self.page.on_keyboard_event = self._handle_keyboard_event
        
    def _handle_keyboard_event(self, e: ft.KeyboardEvent):
        """处理键盘事件"""
        if e.key == "z" and e.ctrl:
            if e.shift:
                self.callbacks['edit_redo'] and self.callbacks['edit_redo']()
            else:
                self.callbacks['edit_undo'] and self.callbacks['edit_undo']()
        elif e.key == "y" and e.ctrl:
            self.callbacks['edit_redo'] and self.callbacks['edit_redo']()
        elif e.key == "s" and e.ctrl:
            if e.shift:
                self.callbacks['file_save_as'] and self.callbacks['file_save_as']()
            else:
                self.callbacks['file_save'] and self.callbacks['file_save']()
        elif e.key == "n" and e.ctrl:
            self.callbacks['file_new'] and self.callbacks['file_new']()
        elif e.key == "o" and e.ctrl:
            self.callbacks['file_open'] and self.callbacks['file_open']()
            
    def _on_recent_files_update(self, files: List[str]):
        """处理最近文件列表更新"""
        self.menu_state.update(recent_files=files)
        self.update()
        
    def update(self):
        """更新菜单状态"""
        self.container.update()

def MenuDemo(page: ft.Page):
    """菜单预览演示"""
    
    # 状态显示区
    status_text = ft.Text("菜单操作状态将显示在这里")
    
    def update_status(action: str):
        status_text.value = f"执行操作: {action}"
        page.update()
    
    # 创建菜单
    menu = Menu(
        page=page,
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
        on_language_settings=lambda: update_status("语言设置"),
        on_theme_settings=lambda: update_status("主题设置"),
        on_auto_save_settings=lambda: update_status("自动保存设置"),
        on_help_documentation=lambda: update_status("查看文档"),
        on_help_feedback=lambda: update_status("提交反馈"),
        on_help_about=lambda: update_status("关于")
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
            menu.container,
            content
        ], spacing=0, expand=True)
    )

if __name__ == "__main__":
    ft.app(target=MenuDemo, view=ft.WEB_BROWSER, port=8080)
