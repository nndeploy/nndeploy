"""
编辑菜单模块

负责:
- 提供编辑相关操作菜单
- 处理撤销、重做、复制、粘贴等功能
- 管理编辑操作状态
- 提供编辑快捷键支持

不可用选项显示为灰色,当前可执行操作高亮显示
"""

from typing import Optional, Callable
import flet as ft
from ...config.language import get_text
from ...config.shortcuts import get_shortcut

class EditMenu(ft.UserControl):
    """编辑菜单"""
    
    def __init__(
        self,
        on_undo: Optional[Callable[[], None]] = None,
        on_redo: Optional[Callable[[], None]] = None,
        on_cut: Optional[Callable[[], None]] = None,
        on_copy: Optional[Callable[[], None]] = None,
        on_paste: Optional[Callable[[], None]] = None,
        on_delete: Optional[Callable[[], None]] = None,
        on_select_all: Optional[Callable[[], None]] = None
    ):
        super().__init__()
        self.on_undo = on_undo
        self.on_redo = on_redo
        self.on_cut = on_cut
        self.on_copy = on_copy
        self.on_paste = on_paste
        self.on_delete = on_delete
        self.on_select_all = on_select_all
        
        # 操作状态
        self._can_undo = False
        self._can_redo = False
        self._has_selection = False
        self._has_clipboard = False
        
    def build(self):
        return ft.PopupMenuButton(
            content=ft.Text(get_text("menu.edit")),
            items=[
                # 撤销
                ft.PopupMenuItem(
                    text=get_text("menu.edit.undo"),
                    icon=ft.icons.UNDO,
                    on_click=lambda _: self.on_undo and self.on_undo(),
                    disabled=not self._can_undo,
                    shortcut=str(get_shortcut("undo"))
                ),
                
                # 重做
                ft.PopupMenuItem(
                    text=get_text("menu.edit.redo"),
                    icon=ft.icons.REDO,
                    on_click=lambda _: self.on_redo and self.on_redo(),
                    disabled=not self._can_redo,
                    shortcut=str(get_shortcut("redo"))
                ),
                
                ft.PopupMenuDivider(),
                
                # 剪切
                ft.PopupMenuItem(
                    text=get_text("menu.edit.cut"),
                    icon=ft.icons.CONTENT_CUT,
                    on_click=lambda _: self.on_cut and self.on_cut(),
                    disabled=not self._has_selection,
                    shortcut=str(get_shortcut("cut"))
                ),
                
                # 复制
                ft.PopupMenuItem(
                    text=get_text("menu.edit.copy"),
                    icon=ft.icons.CONTENT_COPY,
                    on_click=lambda _: self.on_copy and self.on_copy(),
                    disabled=not self._has_selection,
                    shortcut=str(get_shortcut("copy"))
                ),
                
                # 粘贴
                ft.PopupMenuItem(
                    text=get_text("menu.edit.paste"),
                    icon=ft.icons.CONTENT_PASTE,
                    on_click=lambda _: self.on_paste and self.on_paste(),
                    disabled=not self._has_clipboard,
                    shortcut=str(get_shortcut("paste"))
                ),
                
                ft.PopupMenuDivider(),
                
                # 删除
                ft.PopupMenuItem(
                    text=get_text("menu.edit.delete"),
                    icon=ft.icons.DELETE,
                    on_click=lambda _: self.on_delete and self.on_delete(),
                    disabled=not self._has_selection,
                    shortcut=str(get_shortcut("delete"))
                ),
                
                # 全选
                ft.PopupMenuItem(
                    text=get_text("menu.edit.selectAll"),
                    icon=ft.icons.SELECT_ALL,
                    on_click=lambda _: self.on_select_all and self.on_select_all(),
                    shortcut=str(get_shortcut("select_all"))
                ),
            ]
        )
        
    def update_state(
        self,
        can_undo: bool = None,
        can_redo: bool = None,
        has_selection: bool = None,
        has_clipboard: bool = None
    ):
        """更新操作状态
        
        Args:
            can_undo: 是否可以撤销
            can_redo: 是否可以重做
            has_selection: 是否有选中内容
            has_clipboard: 是否有剪贴板内容
        """
        if can_undo is not None:
            self._can_undo = can_undo
        if can_redo is not None:
            self._can_redo = can_redo
        if has_selection is not None:
            self._has_selection = has_selection
        if has_clipboard is not None:
            self._has_clipboard = has_clipboard
            
        self.update() 