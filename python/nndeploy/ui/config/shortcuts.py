"""
快捷键配置模块

负责:
- 定义所有操作的快捷键映射
- 支持自定义快捷键设置
- 管理快捷键冲突
- 提供快捷键查询接口

采用字典形式定义映射关系,支持在UI中展示和修改
"""

from enum import Enum
from typing import Dict, List, Optional, Callable
import json
import os

class ModifierKey(Enum):
    """修饰键枚举"""
    CTRL = "ctrl"
    SHIFT = "shift"
    ALT = "alt"
    META = "meta"  # Windows键或Command键

class ShortcutCategory(Enum):
    """快捷键分类"""
    FILE = "file"          # 文件操作
    EDIT = "edit"          # 编辑操作
    VIEW = "view"          # 视图操作
    NODE = "node"          # 节点操作
    WORKFLOW = "workflow"  # 工作流操作

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
        self.key = key.upper()  # 转换为大写
        self.modifiers = modifiers
        self.category = category
        self.description = description
        self.callback = callback
        
    def __str__(self) -> str:
        """返回快捷键的字符串表示"""
        mods = "+".join(m.value for m in sorted(self.modifiers))
        return f"{mods}+{self.key}" if mods else self.key
        
    def matches(self, key: str, modifiers: List[ModifierKey]) -> bool:
        """检查是否匹配指定的按键组合"""
        return (
            key.upper() == self.key and
            set(modifiers) == set(self.modifiers)
        )

class ShortcutConfig:
    """快捷键配置类"""
    
    def __init__(self):
        self._shortcuts: Dict[str, Shortcut] = {}
        self._load_defaults()
        self._load_custom_shortcuts()
        
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
            "delete": Shortcut(
                key="DELETE",
                modifiers=[],
                category=ShortcutCategory.EDIT,
                description="删除"
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
            
            # 节点操作
            "add_node": Shortcut(
                key="A",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.NODE,
                description="添加节点"
            ),
            "connect_nodes": Shortcut(
                key="L",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.NODE,
                description="连接节点"
            ),
            
            # 工作流操作
            "run_workflow": Shortcut(
                key="R",
                modifiers=[ModifierKey.CTRL],
                category=ShortcutCategory.WORKFLOW,
                description="运行工作流"
            ),
            "stop_workflow": Shortcut(
                key="ESCAPE",
                modifiers=[],
                category=ShortcutCategory.WORKFLOW,
                description="停止工作流"
            ),
        }
        
    def _load_custom_shortcuts(self):
        """从配置文件加载自定义快捷键"""
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
        """保存自定义快捷键配置"""
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../config/custom_shortcuts.json"
        )
        
        custom_config = {
            shortcut_id: {
                "key": shortcut.key,
                "modifiers": [m.value for m in shortcut.modifiers]
            }
            for shortcut_id, shortcut in self._shortcuts.items()
        }
        
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(custom_config, f, indent=4)
        except Exception as e:
            print(f"保存自定义快捷键配置失败: {e}")
            
    def get_shortcut(self, shortcut_id: str) -> Optional[Shortcut]:
        """获取指定ID的快捷键"""
        return self._shortcuts.get(shortcut_id)
        
    def get_shortcuts_by_category(self, category: ShortcutCategory) -> List[Shortcut]:
        """获取指定分类的所有快捷键"""
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
            
        return True
        
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

# 创建全局快捷键配置实例
shortcut_config = ShortcutConfig()

# 便捷函数
def get_shortcut(shortcut_id: str) -> Optional[Shortcut]:
    return shortcut_config.get_shortcut(shortcut_id)

def get_shortcuts_by_category(category: ShortcutCategory) -> List[Shortcut]:
    return shortcut_config.get_shortcuts_by_category(category) 