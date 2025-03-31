"""
快捷键配置模块

负责:
- 定义所有操作的快捷键映射
- 支持自定义快捷键设置
- 管理快捷键冲突
- 提供快捷键查询接口
- 与Flet框架集成处理键盘事件

采用字典形式定义映射关系,支持在UI中展示和修改

暂时不使用
"""

from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import json
import os
import flet as ft
from pathlib import Path

# TODO
from nndeploy.ui.config.language import get_text

class ModifierKey(Enum):
    """修饰键枚举"""
    CTRL = "ctrl"   # Ctrl键
    SHIFT = "shift" # Shift键
    ALT = "alt"     # Alt键
    META = "meta"   # Windows键或Command键

class ShortcutCategory(Enum):
    """快捷键分类"""
    FILE = "file"          # 文件操作
    EDIT = "edit"          # 编辑操作
    VIEW = "view"          # 视图操作
    NODE = "node"          # 节点操作
    WORKFLOW = "workflow"  # 工作流操作
    NAVIGATION = "navigation"  # 导航操作
    TOOLS = "tools"        # 工具操作
    HELP = "help"          # 帮助操作

class Shortcut:
    """快捷键定义类"""
    
    def __init__(
        self,
        key: str,
        modifiers: List[ModifierKey],
        category: ShortcutCategory,
        description: str,
        callback: Optional[Callable] = None
    ):
        """初始化快捷键
        
        Args:
            key: 主按键
            modifiers: 修饰键列表
            category: 快捷键分类
            description: 快捷键描述
            callback: 触发时的回调函数
        """
        self.key = key.upper()  # 转换为大写
        self.modifiers = modifiers
        self.category = category
        self.description = description
        self.callback = callback
        
    def __str__(self) -> str:
        """返回快捷键的字符串表示
        
        Returns:
            格式化的快捷键字符串，如"Ctrl+C"
        """
        mods = "+".join(m.value for m in sorted(self.modifiers, key=lambda m: m.name))
        return f"{mods}+{self.key}" if mods else self.key
        
    def matches(self, key: str, modifiers: List[ModifierKey]) -> bool:
        """检查是否匹配指定的按键组合
        
        Args:
            key: 要检查的主按键
            modifiers: 要检查的修饰键列表
            
        Returns:
            是否匹配
        """
        return (
            key.upper() == self.key and
            set(modifiers) == set(self.modifiers)
        )
    
    def to_flet_key_event(self) -> Dict[str, Any]:
        """转换为Flet键盘事件格式
        
        Returns:
            Flet键盘事件字典
        """
        return {
            "key": self.key,
            "ctrl": ModifierKey.CTRL in self.modifiers,
            "shift": ModifierKey.SHIFT in self.modifiers,
            "alt": ModifierKey.ALT in self.modifiers,
            "meta": ModifierKey.META in self.modifiers
        }

class ShortcutConfig:
    """快捷键配置类"""
    
    def __init__(self):
        """初始化快捷键配置"""
        self._shortcuts: Dict[str, Shortcut] = {}
        self._page: Optional[ft.Page] = None
        self._load_defaults()
        self._load_custom_shortcuts()
        
    def get_shortcuts(self) -> Dict[str, Shortcut]:
        """获取所有快捷键配置"""
        return self._shortcuts
        
    def _load_defaults(self):
        """加载默认快捷键配置"""
        self._shortcuts = {
            # 文件操作
            "new_workflow": Shortcut(
                key="N",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.FILE,
                description="新建工作流"
            ),
            "open_workflow": Shortcut(
                key="O",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.FILE,
                description="打开工作流"
            ),
            "save_workflow": Shortcut(
                key="S",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.FILE,
                description="保存工作流"
            ),
            "save_as": Shortcut(
                key="S",
                modifiers=[ModifierKey.CTRL, ModifierKey.SHIFT],
                category=ShortcutCategory.FILE,
                description="另存为"
            ),
            "export_workflow": Shortcut(
                key="E",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.FILE,
                description="导出工作流"
            ),
            "import_workflow": Shortcut(
                key="I",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.FILE,
                description="导入工作流"
            ),
            "close_workflow": Shortcut(
                key="W",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.FILE,
                description="关闭工作流"
            ),
            "exit_application": Shortcut(
                key="Q",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.FILE,
                description="退出应用"
            ),
            
            # 编辑操作
            "undo": Shortcut(
                key="Z",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.EDIT,
                description="撤销"
            ),
            "redo": Shortcut(
                key="Y",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.EDIT,
                description="重做"
            ),
            "copy": Shortcut(
                key="C",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.EDIT,
                description="复制"
            ),
            "paste": Shortcut(
                key="V",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.EDIT,
                description="粘贴"
            ),
            "cut": Shortcut(
                key="X",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.EDIT,
                description="剪切"
            ),
            "delete": Shortcut(
                key="DELETE",
                modifiers=[],
                category=ShortcutCategory.EDIT,
                description="删除"
            ),
            "select_all": Shortcut(
                key="A",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.EDIT,
                description="全选"
            ),
            "find": Shortcut(
                key="F",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.EDIT,
                description="查找"
            ),
            "replace": Shortcut(
                key="H",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.EDIT,
                description="替换"
            ),
            
            # 视图操作
            "zoom_in": Shortcut(
                key="PLUS",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.VIEW,
                description="放大"
            ),
            "zoom_out": Shortcut(
                key="MINUS",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.VIEW,
                description="缩小"
            ),
            "fit_content": Shortcut(
                key="0",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.VIEW,
                description="适应内容"
            ),
            "toggle_fullscreen": Shortcut(
                key="F11",
                modifiers=[],
                category=ShortcutCategory.VIEW,
                description="切换全屏"
            ),
            "toggle_sidebar": Shortcut(
                key="B",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.VIEW,
                description="切换侧边栏"
            ),
            "toggle_properties": Shortcut(
                key="P",
                modifiers=[ModifierKey.CTRL, ModifierKey.SHIFT],
                category=ShortcutCategory.VIEW,
                description="切换属性面板"
            ),
            "toggle_console": Shortcut(
                key="J",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.VIEW,
                description="切换控制台"
            ),
            
            # 节点操作
            "add_node": Shortcut(
                key="N",
                modifiers=[ModifierKey.ALT],
                category=ShortcutCategory.NODE,
                description="添加节点"
            ),
            "connect_nodes": Shortcut(
                key="L",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.NODE,
                description="连接节点"
            ),
            "disconnect_nodes": Shortcut(
                key="D",
                modifiers=[ModifierKey.CTRL, ModifierKey.SHIFT],
                category=ShortcutCategory.NODE,
                description="断开连接"
            ),
            "group_nodes": Shortcut(
                key="G",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.NODE,
                description="组合节点"
            ),
            "ungroup_nodes": Shortcut(
                key="G",
                modifiers=[ModifierKey.CTRL, ModifierKey.SHIFT],
                category=ShortcutCategory.NODE,
                description="取消组合"
            ),
            "align_nodes_left": Shortcut(
                key="LEFT",
                modifiers=[ModifierKey.CTRL, ModifierKey.ALT],
                category=ShortcutCategory.NODE,
                description="左对齐"
            ),
            "align_nodes_right": Shortcut(
                key="RIGHT",
                modifiers=[ModifierKey.CTRL, ModifierKey.ALT],
                category=ShortcutCategory.NODE,
                description="右对齐"
            ),
            "align_nodes_top": Shortcut(
                key="UP",
                modifiers=[ModifierKey.CTRL, ModifierKey.ALT],
                category=ShortcutCategory.NODE,
                description="顶部对齐"
            ),
            "align_nodes_bottom": Shortcut(
                key="DOWN",
                modifiers=[ModifierKey.CTRL, ModifierKey.ALT],
                category=ShortcutCategory.NODE,
                description="底部对齐"
            ),
            "distribute_nodes_horizontally": Shortcut(
                key="H",
                modifiers=[ModifierKey.CTRL, ModifierKey.ALT],
                category=ShortcutCategory.NODE,
                description="水平分布"
            ),
            "distribute_nodes_vertically": Shortcut(
                key="V",
                modifiers=[ModifierKey.CTRL, ModifierKey.ALT],
                category=ShortcutCategory.NODE,
                description="垂直分布"
            ),
            
            # 工作流操作
            "run_workflow": Shortcut(
                key="F5",
                modifiers=[],
                category=ShortcutCategory.WORKFLOW,
                description="运行工作流"
            ),
            "run_selected_nodes": Shortcut(
                key="F6",
                modifiers=[],
                category=ShortcutCategory.WORKFLOW,
                description="运行选中节点"
            ),
            "stop_workflow": Shortcut(
                key="F7",
                modifiers=[],
                category=ShortcutCategory.WORKFLOW,
                description="停止工作流"
            ),
            "pause_workflow": Shortcut(
                key="F8",
                modifiers=[],
                category=ShortcutCategory.WORKFLOW,
                description="暂停工作流"
            ),
            "resume_workflow": Shortcut(
                key="F9",
                modifiers=[],
                category=ShortcutCategory.WORKFLOW,
                description="恢复工作流"
            ),
            "validate_workflow": Shortcut(
                key="F4",
                modifiers=[],
                category=ShortcutCategory.WORKFLOW,
                description="验证工作流"
            ),
            "optimize_workflow": Shortcut(
                key="O",
                modifiers=[ModifierKey.CTRL, ModifierKey.SHIFT],
                category=ShortcutCategory.WORKFLOW,
                description="优化工作流"
            ),
            
            # 导航操作
            "navigate_back": Shortcut(
                key="LEFT",
                modifiers=[ModifierKey.ALT],
                category=ShortcutCategory.NAVIGATION,
                description="返回上一步"
            ),
            "navigate_forward": Shortcut(
                key="RIGHT",
                modifiers=[ModifierKey.ALT],
                category=ShortcutCategory.NAVIGATION,
                description="前进下一步"
            ),
            "go_to_node": Shortcut(
                key="G",
                modifiers=[ModifierKey.CTRL, ModifierKey.ALT],
                category=ShortcutCategory.NAVIGATION,
                description="跳转到节点"
            ),
            "switch_tab_left": Shortcut(
                key="PAGE_UP",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.NAVIGATION,
                description="切换到左侧标签"
            ),
            "switch_tab_right": Shortcut(
                key="PAGE_DOWN",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.NAVIGATION,
                description="切换到右侧标签"
            ),
            
            # 工具操作
            "open_settings": Shortcut(
                key=",",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.TOOLS,
                description="打开设置"
            ),
            "open_shortcuts": Shortcut(
                key="K",
                modifiers=[ModifierKey.CTRL, ModifierKey.ALT],
                category=ShortcutCategory.TOOLS,
                description="打开快捷键设置"
            ),
            "open_plugins": Shortcut(
                key="P",
                modifiers=[ModifierKey.CTRL, ModifierKey.ALT],
                category=ShortcutCategory.TOOLS,
                description="打开插件管理"
            ),
            "toggle_dark_mode": Shortcut(
                key="D",
                modifiers=[ModifierKey.CTRL, ModifierKey.ALT],
                category=ShortcutCategory.TOOLS,
                description="切换暗色模式"
            ),
            
            # 帮助操作
            "show_help": Shortcut(
                key="F1",
                modifiers=[],
                category=ShortcutCategory.HELP,
                description="显示帮助"
            ),
            "show_about": Shortcut(
                key="F1",
                modifiers=[ModifierKey.SHIFT],
                category=ShortcutCategory.HELP,
                description="关于"
            ),
            "check_updates": Shortcut(
                key="U",
                modifiers=[ModifierKey.CTRL, ModifierKey.ALT],
                category=ShortcutCategory.HELP,
                description="检查更新"
            ),
        }
        
    def _load_custom_shortcuts(self):
        """从配置文件加载自定义快捷键
        
        尝试从自定义配置文件中加载用户定义的快捷键设置，
        如果配置文件存在且格式正确，则覆盖默认快捷键设置。
        """
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../config/custom_shortcuts.json"
        )
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    custom_config = json.load(f)
                    
                for shortcut_id, config in custom_config.items():
                    if shortcut_id in self._shortcuts:
                        shortcut = self._shortcuts[shortcut_id]
                        # 更新快捷键配置
                        shortcut.key = config.get("key", shortcut.key)
                        shortcut.modifiers = [
                            ModifierKey(m) for m in config.get("modifiers", [])
                        ]
            except Exception as e:
                print(f"加载自定义快捷键配置失败: {e}")
                
    def save_custom_shortcuts(self):
        """保存自定义快捷键配置到文件
        
        将当前的快捷键配置保存到自定义配置文件中，
        以便在下次启动应用时恢复用户的自定义设置。
        """
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../config/custom_shortcuts.json"
        )
        
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        custom_config = {
            shortcut_id: {
                "key": shortcut.key,
                "modifiers": [m.value for m in shortcut.modifiers]
            }
            for shortcut_id, shortcut in self._shortcuts.items()
        }
        
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(custom_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"保存自定义快捷键配置失败: {e}")
            
    def get_shortcut(self, shortcut_id: str) -> Optional[Shortcut]:
        """获取指定ID的快捷键
        
        Args:
            shortcut_id: 快捷键ID
            
        Returns:
            对应的快捷键对象，如果不存在则返回None
        """
        return self._shortcuts.get(shortcut_id)
        
    def get_shortcuts_by_category(self, category: ShortcutCategory) -> List[Shortcut]:
        """获取指定分类的所有快捷键
        
        Args:
            category: 快捷键分类
            
        Returns:
            该分类下的所有快捷键列表
        """
        return [
            shortcut for shortcut in self._shortcuts.values()
            if shortcut.category == category
        ]
        
    def set_shortcut(
        self,
        shortcut_id: str,
        key: str,
        modifiers: List[ModifierKey],
        save: bool = True
    ) -> bool:
        """设置快捷键
        
        Args:
            shortcut_id: 快捷键ID
            key: 按键
            modifiers: 修饰键列表
            save: 是否保存到配置文件
            
        Returns:
            是否设置成功
        """
        if shortcut_id not in self._shortcuts:
            return False
            
        # 检查是否与其他快捷键冲突
        for other_id, other in self._shortcuts.items():
            if other_id != shortcut_id and other.matches(key, modifiers):
                return False
                
        shortcut = self._shortcuts[shortcut_id]
        shortcut.key = key
        shortcut.modifiers = modifiers
        
        if save:
            self.save_custom_shortcuts()
            
        # 如果已经注册了页面，更新快捷键绑定
        if self._page:
            self._update_flet_keyboard_shortcuts()
            
        return True
    
    def _update_flet_keyboard_shortcuts(self):
        """更新Flet页面的键盘快捷键绑定
        
        当快捷键配置发生变化时，更新Flet页面的键盘事件处理。
        """
        # 这个方法在未来可能需要实现，目前通过全局键盘事件处理器处理
        pass
    
    def register_page(self, page: ft.Page):
        """注册Flet页面以启用快捷键功能
        
        Args:
            page: Flet页面对象
        """
        self._page = page
        
        # 设置全局键盘事件处理器
        def on_keyboard(e: ft.KeyboardEvent):
            # 将Flet键盘事件转换为我们的格式
            key = e.key
            modifiers = []
            if e.ctrl: modifiers.append(ModifierKey.CTRL)
            if e.shift: modifiers.append(ModifierKey.SHIFT)
            if e.alt: modifiers.append(ModifierKey.ALT)
            if e.meta: modifiers.append(ModifierKey.META)
            
            # 处理快捷键
            self.handle_keypress(key, modifiers)
            
        # 注册键盘事件处理器
        page.on_keyboard_event = on_keyboard
    
    def handle_keypress(self, key: str, modifiers: List[ModifierKey]) -> bool:
        """处理按键事件
        
        Args:
            key: 按键
            modifiers: 修饰键列表
            
        Returns:
            是否有快捷键被触发
        """
        for shortcut in self._shortcuts.values():
            if shortcut.matches(key, modifiers) and shortcut.callback:
                shortcut.callback()
                return True
        return False
    
    def set_callback(self, shortcut_id: str, callback: Callable) -> bool:
        """为快捷键设置回调函数
        
        Args:
            shortcut_id: 快捷键ID
            callback: 回调函数
            
        Returns:
            是否设置成功
        """
        if shortcut_id not in self._shortcuts:
            return False
            
        self._shortcuts[shortcut_id].callback = callback
        return True
    
    def create_shortcuts_table(self) -> ft.DataTable:
        """创建快捷键表格UI组件
        
        创建一个展示所有快捷键的表格，按分类组织，
        并提供编辑和重置功能。
        
        Returns:
            快捷键表格组件
        """
        # 按分类组织快捷键
        shortcuts_by_category = {}
        for category in ShortcutCategory:
            shortcuts_by_category[category] = self.get_shortcuts_by_category(category)
        
        # 创建数据表格
        table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text(get_text("shortcuts.category"))),
                ft.DataColumn(ft.Text(get_text("shortcuts.description"))),
                ft.DataColumn(ft.Text(get_text("shortcuts.key_combination"))),
                ft.DataColumn(ft.Text(get_text("shortcuts.actions"))),
            ],
            rows=[]
        )
        
        # 添加行
        for category, shortcuts in shortcuts_by_category.items():
            if not shortcuts:  # 跳过没有快捷键的分类
                continue
                
            # 添加分类标题行
            table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(get_text(f"shortcuts.category.{category.value}"), weight=ft.FontWeight.BOLD)),
                        ft.DataCell(ft.Text("")),
                        ft.DataCell(ft.Text("")),
                        ft.DataCell(ft.Text("")),
                    ],
                    color=ft.colors.BLUE_50
                )
            )
            
            # 添加快捷键行
            for shortcut in shortcuts:
                table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text("")),
                            ft.DataCell(ft.Text(get_text(f"shortcuts.{shortcut.description}"))),
                            ft.DataCell(ft.Text(str(shortcut))),
                            ft.DataCell(
                                ft.Row([
                                    ft.IconButton(
                                        icon=ft.icons.EDIT,
                                        tooltip=get_text("shortcuts.edit"),
                                        on_click=lambda e, s=shortcut: self._show_edit_dialog(s)
                                    ),
                                    ft.IconButton(
                                        icon=ft.icons.RESTORE,
                                        tooltip=get_text("shortcuts.reset"),
                                        on_click=lambda e, s=shortcut: self._reset_shortcut(s)
                                    )
                                ])
                            ),
                        ]
                    )
                )
        
        return table
    
    def _reset_shortcut(self, shortcut: Shortcut):
        """重置快捷键为默认值
        
        Args:
            shortcut: 要重置的快捷键
        """
        if not self._page:
            return
            
        # 查找快捷键ID
        shortcut_id = None
        for sid, s in self._shortcuts.items():
            if s == shortcut:
                shortcut_id = sid
                break
                
        if not shortcut_id:
            return
            
        # 重新加载默认配置中的这个快捷键
        temp_config = ShortcutConfig()
        temp_config._load_defaults()
        default_shortcut = temp_config.get_shortcut(shortcut_id)
        
        if not default_shortcut:
            return
            
        # 设置为默认值
        success = self.set_shortcut(
            shortcut_id, 
            default_shortcut.key, 
            default_shortcut.modifiers
        )
        
        # 显示结果通知
        if success:
            self._page.snack_bar = ft.SnackBar(
                content=ft.Text(get_text("shortcuts.reset_success")),
                bgcolor=ft.colors.GREEN
            )
        else:
            self._page.snack_bar = ft.SnackBar(
                content=ft.Text(get_text("shortcuts.reset_failed")),
                bgcolor=ft.colors.RED
            )
            
        self._page.snack_bar.open = True
        self._page.update()
    
    def _show_edit_dialog(self, shortcut: Shortcut):
        """显示编辑快捷键对话框
        
        Args:
            shortcut: 要编辑的快捷键
        """
        if not self._page:
            return
        
        # 创建一个临时变量来存储新的快捷键
        current_key = shortcut.key
        current_modifiers = shortcut.modifiers.copy()
        shortcut_id = None
        
        # 查找快捷键ID
        for sid, s in self._shortcuts.items():
            if s == shortcut:
                shortcut_id = sid
                break
        
        if not shortcut_id:
            return
            
        # 创建显示当前快捷键的文本框
        key_display = ft.TextField(
            value=str(shortcut),
            read_only=True,
            label=get_text("shortcuts.current_combination")
        )
        
        # 创建状态文本
        status_text = ft.Text("", color=ft.colors.RED)
        
        # 创建键盘监听函数
        def on_key_capture(e: ft.KeyboardEvent):
            nonlocal current_key, current_modifiers
            
            # 忽略某些特殊键
            if e.key in ["Tab", "Escape"]:
                return
                
            # 收集修饰键
            new_modifiers = []
            if e.ctrl: new_modifiers.append(ModifierKey.CTRL)
            if e.shift: new_modifiers.append(ModifierKey.SHIFT)
            if e.alt: new_modifiers.append(ModifierKey.ALT)
            if e.meta: new_modifiers.append(ModifierKey.META)
            
            # 设置新的按键和修饰键
            current_key = e.key.upper()
            current_modifiers = new_modifiers
            
            # 检查冲突
            has_conflict = False
            conflict_with = ""
            for other_id, other in self._shortcuts.items():
                if other_id != shortcut_id and other.matches(current_key, current_modifiers):
                    has_conflict = True
                    conflict_with = get_text(f"shortcuts.{other.description}")
                    break
            
            # 更新显示
            temp_shortcut = Shortcut(
                current_key, 
                current_modifiers,
                shortcut.category,
                shortcut.description
            )
            key_display.value = str(temp_shortcut)
            
            if has_conflict:
                status_text.value = get_text("shortcuts.conflict").format(conflict_with)
                status_text.color = ft.colors.RED
            else:
                status_text.value = get_text("shortcuts.valid_combination")
                status_text.color = ft.colors.GREEN
                
            self._page.update()
            
        # 创建键盘捕获区域
        key_capture = ft.TextField(
            label=get_text("shortcuts.press_key_combination"),
            autofocus=True,
            on_key_event=on_key_capture,
            focused_border_color=ft.colors.BLUE,
            border_color=ft.colors.BLUE_200
        )
            
        # 创建对话框
        dialog = ft.AlertDialog(
            title=ft.Text(get_text("shortcuts.edit_title")),
            content=ft.Column(
                [
                    ft.Text(get_text("shortcuts.edit_instruction")),
                    key_display,
                    key_capture,
                    status_text,
                ],
                width=400,
                height=200,
                spacing=20
            ),
            actions=[
                ft.TextButton(
                    get_text("dialog.cancel"), 
                    on_click=lambda e: self._close_edit_dialog()
                ),
                ft.TextButton(
                    get_text("dialog.confirm"), 
                    on_click=lambda e: self._save_shortcut_edit(shortcut_id, current_key, current_modifiers)
                ),
            ],
        )
        
        # 显示对话框
        self._page.dialog = dialog
        self._page.dialog.open = True
        self._page.update()
        
    def _close_edit_dialog(self):
        """关闭编辑快捷键对话框"""
        if self._page:
            self._page.dialog.open = False
            self._page.update()
    
    def _save_shortcut_edit(self, shortcut_id: str, key: str, modifiers: List[ModifierKey]):
        """保存编辑的快捷键
        
        Args:
            shortcut_id: 快捷键ID
            key: 新的按键
            modifiers: 新的修饰键列表
        """
        if not self._page:
            return
            
        # 设置新的快捷键
        success = self.set_shortcut(shortcut_id, key, modifiers)
        
        # 关闭对话框
        self._page.dialog.open = False
        
        # 显示结果通知
        if success:
            self._page.snack_bar = ft.SnackBar(
                content=ft.Text(get_text("shortcuts.save_success")),
                bgcolor=ft.colors.GREEN
            )
        else:
            self._page.snack_bar = ft.SnackBar(
                content=ft.Text(get_text("shortcuts.save_failed")),
                bgcolor=ft.colors.RED
            )
            
        self._page.snack_bar.open = True
        self._page.update()

# 创建全局快捷键配置实例
shortcut_config = ShortcutConfig()

# 便捷函数
def get_shortcut(shortcut_id: str) -> Optional[Shortcut]:
    return shortcut_config.get_shortcut(shortcut_id)

def get_shortcuts_by_category(category: ShortcutCategory) -> List[Shortcut]:
    return shortcut_config.get_shortcuts_by_category(category)

def register_page(page: ft.Page):
    """注册Flet页面以启用快捷键功能
    
    Args:
        page: Flet页面对象
    """
    shortcut_config.register_page(page)

def set_callback(shortcut_id: str, callback: Callable) -> bool:
    """为快捷键设置回调函数
    
    Args:
        shortcut_id: 快捷键ID
        callback: 回调函数
        
    Returns:
        是否设置成功
    """
    return shortcut_config.set_callback(shortcut_id, callback)

def create_shortcuts_table() -> ft.DataTable:
    """创建快捷键表格UI组件
    
    Returns:
        快捷键表格组件
    """
    return shortcut_config.create_shortcuts_table()


def main(page: ft.Page):
    """快捷键配置预览界面
    
    Args:
        page: Flet页面对象
    """
    page.title = "快捷键配置预览"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    
    # 注册页面以启用快捷键功能
    register_page(page)
    
    # 创建测试回调函数
    def test_callback(e=None):
        page.snack_bar = ft.SnackBar(
            content=ft.Text(f"触发了快捷键: {e.shortcut if e else '未知'}"),
            bgcolor=ft.colors.BLUE
        )
        page.snack_bar.open = True
        page.update()
    
    # 为一些快捷键设置测试回调
    # 获取快捷键字典并为前5个设置测试回调
    shortcuts_dict = shortcut_config.get_shortcuts()
    for i, (shortcut_id, shortcut) in enumerate(shortcuts_dict.items()):
        if i < 100:  # 只为前5个快捷键设置回调
            set_callback(shortcut_id, test_callback)
    
    # 创建快捷键表格
    shortcuts_table = create_shortcuts_table()
    
    # 添加说明文本
    instructions = ft.Column([
        ft.Text("快捷键配置预览", size=24, weight=ft.FontWeight.BOLD),
        ft.Text("此页面展示了应用中定义的所有快捷键，可以查看和测试快捷键功能。"),
        ft.Text("前5个快捷键已设置测试回调，按下对应快捷键将显示提示信息。"),
        ft.Text("提示: 可以点击表格中的快捷键组合进行编辑。", italic=True),
    ])
    
    # 构建页面
    page.add(
        instructions,
        ft.Divider(),
        shortcuts_table,
    )
    
# 如果直接运行此文件，则启动快捷键预览
if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=8080)
