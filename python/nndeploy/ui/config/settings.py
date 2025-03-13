"""
全局设置模块

负责:
- 管理应用程序的全局配置参数
- 提供配置的读写接口
- 处理配置的持久化
- 支持配置的导入导出

配置项使用字典或类的形式定义,支持从配置文件加载
"""

from typing import Any, Dict, Optional
import json
import os
from pathlib import Path

class Settings:
    """全局设置类"""
    
    # 默认设置
    DEFAULTS = {
        # 画布设置
        "canvas": {
            "width": 1920,          # 默认宽度
            "height": 1080,         # 默认高度
            "grid_size": 20,        # 网格大小
            "grid_enabled": True,   # 是否显示网格
            "grid_color": "#E5E5E5",# 网格颜色
            "snap_to_grid": True,   # 是否对齐到网格
            "zoom_min": 0.1,        # 最小缩放比例
            "zoom_max": 5.0,        # 最大缩放比例
            "zoom_step": 0.1,       # 缩放步长
        },
        
        # 节点设置
        "node": {
            "default_width": 200,   # 默认宽度
            "default_height": 100,  # 默认高度
            "min_width": 100,       # 最小宽度
            "min_height": 50,       # 最小高度
            "padding": 10,          # 内边距
            "border_radius": 6,     # 边框圆角
            "font_size": 14,        # 字体大小
            "line_height": 1.5,     # 行高
        },
        
        # 连线设置
        "edge": {
            "line_width": 2,        # 线条宽度
            "line_color": "#666666",# 线条颜色
            "arrow_size": 8,        # 箭头大小
            "curve_factor": 0.5,    # 曲线因子
            "snap_distance": 10,    # 吸附距离
        },
        
        # 自动保存设置
        "auto_save": {
            "enabled": True,        # 是否启用
            "interval": 300,        # 间隔(秒)
            "max_backups": 5,       # 最大备份数
        },
        
        # 性能设置
        "performance": {
            "render_quality": "high",    # 渲染质量(low/medium/high)
            "animation_enabled": True,    # 是否启用动画
            "cache_size": 100,           # 缓存大小(MB)
            "max_undo_steps": 50,        # 最大撤销步数
        },
        
        # 界面设置
        "ui": {
            "sidebar_width": 300,        # 侧边栏宽度
            "panel_width": 400,          # 面板宽度
            "toolbar_position": "top",   # 工具栏位置
            "show_status_bar": True,     # 显示状态栏
            "show_minimap": True,        # 显示小地图
        }
    }
    
    def __init__(self):
        self._settings = self.DEFAULTS.copy()
        self._observers = []
        self._load_settings()
        
    def _load_settings(self):
        """从配置文件加载设置"""
        config_path = self._get_config_path()
        
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    saved_settings = json.load(f)
                    # 递归更新设置,保留默认值
                    self._update_dict(self._settings, saved_settings)
            except Exception as e:
                print(f"加载设置失败: {e}")
                
    def _save_settings(self):
        """保存设置到配置文件"""
        config_path = self._get_config_path()
        
        # 确保配置目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4)
        except Exception as e:
            print(f"保存设置失败: {e}")
            
    def _get_config_path(self) -> Path:
        """获取配置文件路径"""
        return Path(os.path.dirname(os.path.abspath(__file__))) / "../config/settings.json"
        
    def _update_dict(self, target: Dict, source: Dict):
        """递归更新字典,保留目标字典中的键"""
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    self._update_dict(target[key], value)
                else:
                    target[key] = value
                    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """获取设置值
        
        Args:
            section: 设置分类
            key: 设置键名
            default: 默认值
            
        Returns:
            设置值
        """
        try:
            return self._settings[section][key]
        except KeyError:
            return default
            
    def set(self, section: str, key: str, value: Any, save: bool = True):
        """设置值
        
        Args:
            section: 设置分类
            key: 设置键名
            value: 设置值
            save: 是否保存到文件
        """
        try:
            if self._settings[section][key] != value:
                self._settings[section][key] = value
                # 通知观察者
                for observer in self._observers:
                    observer(section, key, value)
                # 保存到文件
                if save:
                    self._save_settings()
        except KeyError:
            pass
            
    def get_section(self, section: str) -> Dict:
        """获取整个分类的设置
        
        Args:
            section: 设置分类
            
        Returns:
            分类设置字典
        """
        return self._settings.get(section, {}).copy()
        
    def reset_section(self, section: str, save: bool = True):
        """重置分类设置为默认值
        
        Args:
            section: 设置分类
            save: 是否保存到文件
        """
        if section in self._settings and section in self.DEFAULTS:
            self._settings[section] = self.DEFAULTS[section].copy()
            # 通知观察者
            for observer in self._observers:
                observer(section, None, self._settings[section])
            # 保存到文件
            if save:
                self._save_settings()
                
    def reset_all(self, save: bool = True):
        """重置所有设置为默认值
        
        Args:
            save: 是否保存到文件
        """
        self._settings = self.DEFAULTS.copy()
        # 通知观察者
        for observer in self._observers:
            observer(None, None, self._settings)
        # 保存到文件
        if save:
            self._save_settings()
            
    def add_observer(self, observer):
        """添加设置变化观察者"""
        if observer not in self._observers:
            self._observers.append(observer)
            
    def remove_observer(self, observer):
        """移除设置变化观察者"""
        if observer in self._observers:
            self._observers.remove(observer)
            
    def export_settings(self, filepath: str):
        """导出设置到文件
        
        Args:
            filepath: 导出文件路径
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4)
        except Exception as e:
            print(f"导出设置失败: {e}")
            
    def import_settings(self, filepath: str, save: bool = True):
        """从文件导入设置
        
        Args:
            filepath: 导入文件路径
            save: 是否保存到配置文件
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                imported_settings = json.load(f)
                # 递归更新设置
                self._update_dict(self._settings, imported_settings)
                # 通知观察者
                for observer in self._observers:
                    observer(None, None, self._settings)
                # 保存到文件
                if save:
                    self._save_settings()
        except Exception as e:
            print(f"导入设置失败: {e}")

# 创建全局设置实例
settings = Settings()

# 便捷函数
def get_setting(section: str, key: str, default: Any = None) -> Any:
    return settings.get(section, key, default)

def set_setting(section: str, key: str, value: Any, save: bool = True):
    settings.set(section, key, value, save) 