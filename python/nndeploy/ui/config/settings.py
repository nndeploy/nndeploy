"""
全局设置模块

职责:
- 管理应用程序范围的配置参数
- 提供读取和写入配置的接口
- 处理配置持久化
- 支持配置导入和导出

配置项以字典或类的形式定义，支持从配置文件加载
"""

from typing import Any, Dict, Optional, Callable, List, Union
import json
import os
from pathlib import Path
import flet as ft

class Settings:
    """
    全局设置管理器
    
    管理所有应用程序配置参数，提供统一的读写接口，
    支持配置持久化、导入/导出和变更通知机制。
    """
    
    # 默认设置
    DEFAULTS = {
        # 画布设置
        "canvas": {
            "width": "auto",        # 自适应宽度
            "height": "auto",       # 自适应高度
            "grid_size": 20,        # 网格大小
            "grid_enabled": True,   # 显示网格
            "grid_color": "#E5E5E5",# 网格颜色 (浅灰色，用于在画布上显示网格线，确保可见但不干扰内容)
            "snap_to_grid": True,   # 对齐网格
            "zoom_min": 0.1,        # 最小缩放级别
            "zoom_max": 5.0,        # 最大缩放级别
            "zoom_step": 0.1,       # 缩放步长
        },
        
        # 节点设置
        "node": {
            "default_width": 200,   # 默认宽度
            "default_height": 100,  # 默认高度
            "min_width": 100,       # 最小宽度
            "min_height": 50,       # 最小高度
            "padding": 10,          # 内部填充
            "border_radius": 6,     # 边框圆角
            "font_size": 14,        # 字体大小
            "line_height": 1.5,     # 行高
        },
        
        # 边缘设置
        "edge": {
            "line_width": 2,        # 线宽
            "line_color": "#666666",# 线条颜色 (中灰色，用于连接节点的线条，提供足够的对比度但不过于显眼)
            "curve_factor": 0.5,    # 曲线因子
            "snap_distance": 10,    # 吸附距离
        },
        
        # 自动保存设置
        "auto_save": {
            "enabled": True,        # 启用自动保存
            "interval": 300,        # 间隔（秒）
            "max_backups": 5,       # 最大备份数
        },
        
        # 性能设置
        "performance": {
            "render_quality": "high",    # 渲染质量（低/中/高）
            "animation_enabled": True,    # 启用动画
            "cache_size": 100,           # 缓存大小（MB）
            "max_undo_steps": 50,        # 最大撤销步骤
        },
        
        # UI设置
        "ui": {
            "panel_width": 400,          # 侧边栏宽度（像素单位，控制左侧导航栏的宽度，影响工作区可用空间）
            "node_config_width": 400,    # 节点配置宽度（像素单位，控制右侧属性面板的宽度，过宽会挤压中央画布区域）
            "toolbar_position": "top",   # 工具栏位置
            "show_status_bar": True,     # 显示状态栏
            "theme": "light",            # 主题模式（亮/暗）
            "language": "zh_CN",         # 语言设置
        }
    }
    
    def __init__(self):
        """初始化设置管理器，加载默认值并从配置文件更新"""
        self._settings: Dict[str, Dict[str, Any]] = self.DEFAULTS.copy() # 二级map
        # 观察者列表，用于实现设置变更通知机制
        # 每个观察者是一个回调函数，当设置发生变化时会被调用
        # 函数签名为：callback(section, key, value)
        #   - section: 变更的设置部分（如"ui"、"node"等），为None时表示整个设置都已更新
        #   - key: 变更的具体设置键，为None时表示整个section都已更新
        #   - value: 新的设置值，可能是单个值或整个设置字典
        # 这种设计采用了观察者模式，允许UI组件监听设置变化并实时响应
        # 比如当主题从"light"变为"dark"时，所有注册的观察者都会收到通知
        # 观察者列表，每个观察者是一个回调函数
        # 函数参数中的None有特殊含义：
        # - 当section为None时，表示整个设置字典都已更新（如导入设置时）
        # - 当key为None时，表示整个section下的所有设置都已更新
        # 例如：
        # - observer(None, None, settings) 表示整个设置都已更新
        # - observer("ui", None, ui_settings) 表示ui部分的所有设置都已更新
        # - observer("ui", "theme", "dark") 表示ui.theme设置已更新为"dark"
        self._observers: List[Callable[[Optional[str], Optional[str], Any], None]] = []
        self._load_settings()
        
    def _load_settings(self) -> None:
        """从配置文件加载设置，如果文件不存在或格式错误则保留默认值"""
        config_path = self._get_config_path()
        
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    saved_settings = json.load(f)
                    # 递归更新设置，同时保留默认结构
                    self._update_dict(self._settings, saved_settings)
            except json.JSONDecodeError as e:
                print(f"配置文件格式错误: {e}")
            except Exception as e:
                print(f"加载设置失败: {e}")
                
    def _save_settings(self) -> None:
        """保存设置到配置文件，确保目录存在"""
        config_path = self._get_config_path()
        
        # 确保配置目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"保存设置失败: {e}")
            
    def _get_config_path(self) -> Path:
        """获取配置文件路径"""
        # 使用更可靠的路径构建
        return Path(os.path.dirname(os.path.abspath(__file__))) / "../config/settings.json"
        
    def _update_dict(self, target: Dict, source: Dict) -> None:
        """
        递归更新字典，同时保留目标结构
        
        参数:
            target: 目标字典
            source: 源字典
        """
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    self._update_dict(target[key], value)
                else:
                    target[key] = value
                    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        获取设置值
        
        参数:
            section: 设置部分
            key: 设置键
            default: 默认值
            
        返回:
            设置值，如果未找到则返回默认值
        """
        try:
            return self._settings[section][key]
        except KeyError:
            return default
            
    def set(self, section: str, key: str, value: Any, save: bool = True) -> bool:
        """
        设置值
        
        参数:
            section: 设置部分
            key: 设置键
            value: 设置值
            save: 是否保存到文件
            
        返回:
            设置是否成功更新
        """
        try:
            if section not in self._settings:
                self._settings[section] = {}
                
            if self._settings[section].get(key) != value:
                self._settings[section][key] = value
                # 通知观察者
                for observer in self._observers:
                    observer(section, key, value)
                # 保存到文件
                if save:
                    self._save_settings()
                return True
            return False
        except Exception:
            return False
            
    def get_section(self, section: str) -> Dict:
        """
        获取设置的整个部分
        
        参数:
            section: 设置部分
            
        返回:
            部分设置字典的副本
        """
        return self._settings.get(section, {}).copy()
        
    def reset_section(self, section: str, save: bool = True) -> bool:
        """
        将部分设置重置为默认值
        
        参数:
            section: 设置部分
            save: 是否保存到文件
            
        返回:
            重置是否成功
        """
        if section in self._settings and section in self.DEFAULTS:
            self._settings[section] = self.DEFAULTS[section].copy()
            # 通知观察者
            for observer in self._observers:
                observer(section, None, self._settings[section])
            # 保存到文件
            if save:
                self._save_settings()
            return True
        return False
                
    def reset_all(self, save: bool = True) -> None:
        """
        将所有设置重置为默认值
        
        参数:
            save: 是否保存到文件
        """
        self._settings = self.DEFAULTS.copy()
        # 通知观察者
        for observer in self._observers:
            observer(None, None, self._settings)
        # 保存到文件
        if save:
            self._save_settings()
            
    def add_observer(self, observer: Callable[[Optional[str], Optional[str], Any], None]) -> None:
        """
        添加设置变更观察者
        
        参数:
            observer: 观察者回调函数，接收(section, key, value)参数
        """
        if observer not in self._observers:
            self._observers.append(observer)
            
    def remove_observer(self, observer: Callable[[Optional[str], Optional[str], Any], None]) -> None:
        """
        移除设置变更观察者
        
        参数:
            observer: 要移除的观察者回调函数
        """
        if observer in self._observers:
            self._observers.remove(observer)
            
    def export_settings(self, filepath: Union[str, Path]) -> bool:
        """
        导出设置到文件
        
        参数:
            filepath: 导出文件路径
            
        返回:
            导出是否成功
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"导出设置失败: {e}")
            return False
            
    def import_settings(self, filepath: Union[str, Path], save: bool = True) -> bool:
        """
        从文件导入设置
        
        参数:
            filepath: 导入文件路径
            save: 是否保存到配置文件
            
        返回:
            导入是否成功
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
                return True
        except json.JSONDecodeError:
            print(f"导入文件格式错误: {filepath}")
            return False
        except Exception as e:
            print(f"导入设置失败: {e}")
            return False

# 创建全局设置实例
settings = Settings()

# 便捷函数
def get_setting(section: str, key: str, default: Any = None) -> Any:
    """
    获取设置值的便捷函数
    
    参数:
        section: 设置部分
        key: 设置键
        default: 默认值
        
    返回:
        设置值
    """
    return settings.get(section, key, default)

def set_setting(section: str, key: str, value: Any, save: bool = True) -> bool:
    """
    设置值的便捷函数
    
    参数:
        section: 设置部分
        key: 设置键
        value: 设置值
        save: 是否保存到文件
        
    返回:
        设置是否成功更新
    """
    return settings.set(section, key, value, save)


############################## 预览演示 ##############################
class SettingsPreview:
    """设置预览演示类"""
    
    def __init__(self):
        self.page = None
        
    def _create_setting_row(self, section: str, key: str, value: Any, value_type: str = "text"):
        """创建设置行"""
        row = ft.Row([
            ft.Text(f"{section}.{key}", width=200),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        
        # 根据值类型创建不同的控件
        if value_type == "bool":
            switch = ft.Switch(value=value)
            
            def on_switch_change(e):
                set_setting(section, key, switch.value)
                
            switch.on_change = on_switch_change
            row.controls.append(switch)
            
        elif value_type == "color":
            color_picker = ft.IconButton(
                icon=ft.Icons.COLOR_LENS,
                icon_color=value,
                tooltip="选择颜色"
            )
            
            def on_color_picked(e: ft.ControlEvent):
                if e.control.data:
                    color = e.control.data
                    set_setting(section, key, color)
                    color_picker.icon_color = color
                    self.page.update()
            
            color_picker.on_click = lambda _: self.page.launch_color_picker(
                on_color_picked, color=value
            )
            
            row.controls.append(color_picker)
            
        elif value_type == "dropdown":
            options = []
            if key == "theme":
                options = ["light", "dark"]
            elif key == "render_quality":
                options = ["low", "medium", "high"]
            
            dropdown = ft.Dropdown(
                value=value,
                options=[ft.dropdown.Option(opt) for opt in options],
                width=150
            )
            
            def on_dropdown_change(e):
                set_setting(section, key, dropdown.value)
                
            dropdown.on_change = on_dropdown_change
            row.controls.append(dropdown)
            
        elif value_type == "number":
            # 获取合适的最大值
            max_value = 100  # 默认最大值
            
            # 根据不同设置项设置合适的最大值
            if section == "canvas":
                if key == "grid_size":
                    max_value = 50
                elif key == "zoom_max":
                    max_value = 10.0
            elif section == "node":
                if key == "default_width":
                    max_value = 500
                elif key == "default_height":
                    max_value = 300
                elif key == "border_radius":
                    max_value = 20
                elif key == "font_size":
                    max_value = 30
            elif section == "performance":
                if key == "max_undo_steps":
                    max_value = 100
            elif section == "ui":
                if key == "panel_width":
                    max_value = 800
            
            # 确保当前值不超过最大值
            current_value = min(value, max_value)
            
            slider = ft.Slider(
                min=0,
                max=max_value,
                value=current_value,
                divisions=20,
                label="{value}"
            )
            
            def on_slider_change(e):
                set_setting(section, key, int(slider.value))
                
            slider.on_change = on_slider_change
            row.controls.append(slider)
            
        else:  # 文本
            text_field = ft.TextField(value=str(value), width=150)
            
            def on_text_change(e):
                set_setting(section, key, text_field.value)
                
            text_field.on_change = on_text_change
            row.controls.append(text_field)
            
        return row
        
    def _build_settings_view(self):
        """构建设置视图"""
        tabs = []
        
        # 画布设置
        canvas_settings = settings.get_section("canvas")
        canvas_view = ft.Column([
            self._create_setting_row("canvas", "grid_size", canvas_settings["grid_size"], "number"),
            self._create_setting_row("canvas", "grid_enabled", canvas_settings["grid_enabled"], "bool"),
            self._create_setting_row("canvas", "grid_color", canvas_settings["grid_color"], "color"),
            self._create_setting_row("canvas", "snap_to_grid", canvas_settings["snap_to_grid"], "bool"),
        ], spacing=10)
        
        tabs.append(ft.Tab(text="画布", content=canvas_view))
        
        # 节点设置
        node_settings = settings.get_section("node")
        node_view = ft.Column([
            self._create_setting_row("node", "default_width", node_settings["default_width"], "number"),
            self._create_setting_row("node", "default_height", node_settings["default_height"], "number"),
            self._create_setting_row("node", "border_radius", node_settings["border_radius"], "number"),
            self._create_setting_row("node", "font_size", node_settings["font_size"], "number"),
        ], spacing=10)
        
        tabs.append(ft.Tab(text="节点", content=node_view))
        
        # UI设置
        ui_settings = settings.get_section("ui")
        ui_view = ft.Column([
            self._create_setting_row("ui", "theme", ui_settings["theme"], "dropdown"),
            self._create_setting_row("ui", "show_status_bar", ui_settings["show_status_bar"], "bool"),
            self._create_setting_row("ui", "panel_width", ui_settings["panel_width"], "number"),
        ], spacing=10)
        
        tabs.append(ft.Tab(text="界面", content=ui_view))
        
        # 性能设置
        perf_settings = settings.get_section("performance")
        perf_view = ft.Column([
            self._create_setting_row("performance", "render_quality", perf_settings["render_quality"], "dropdown"),
            self._create_setting_row("performance", "animation_enabled", perf_settings["animation_enabled"], "bool"),
            self._create_setting_row("performance", "max_undo_steps", perf_settings["max_undo_steps"], "number"),
        ], spacing=10)
        
        tabs.append(ft.Tab(text="性能", content=perf_view))
        
        return ft.Tabs(tabs=tabs, expand=True)
        
    def _build_actions(self):
        """构建操作按钮"""
        return ft.Row([
            ft.ElevatedButton(
                text="重置所有设置",
                icon=ft.Icons.RESTORE,
                on_click=lambda _: self._reset_all()
            ),
            ft.ElevatedButton(
                text="导出设置",
                icon=ft.Icons.DOWNLOAD,
                on_click=lambda _: self._export_settings()
            ),
            ft.ElevatedButton(
                text="导入设置",
                icon=ft.Icons.UPLOAD_FILE,
                on_click=lambda _: self._import_settings()
            ),
        ], alignment=ft.MainAxisAlignment.END)
        
    def _reset_all(self):
        """重置所有设置"""
        settings.reset_all()
        # 刷新UI
        self.page.clean()
        self.page.add(self._build_ui())
        
    def _export_settings(self):
        """导出设置"""
        def on_save_dialog(e: ft.FilePickerResultEvent):
            if e.path:
                if settings.export_settings(e.path):
                    self.page.show_snack_bar(ft.SnackBar(
                        content=ft.Text(f"设置已导出到 {e.path}"),
                        action="确定"
                    ))
                else:
                    self.page.show_snack_bar(ft.SnackBar(
                        content=ft.Text("导出设置失败"),
                        action="确定"
                    ))
                    
        save_dialog = ft.FilePicker(on_result=on_save_dialog)
        self.page.overlay.append(save_dialog)
        self.page.update()
        save_dialog.save_file(
            dialog_title="导出设置",
            file_name="settings.json",
            allowed_extensions=["json"]
        )
        
    def _import_settings(self):
        """导入设置"""
        def on_open_dialog(e: ft.FilePickerResultEvent):
            if e.files and len(e.files) > 0:
                file_path = e.files[0].path
                if settings.import_settings(file_path):
                    self.page.show_snack_bar(ft.SnackBar(
                        content=ft.Text("设置已导入"),
                        action="确定"
                    ))
                    # 刷新UI
                    self.page.clean()
                    self.page.add(self._build_ui())
                else:
                    self.page.show_snack_bar(ft.SnackBar(
                        content=ft.Text("导入设置失败"),
                        action="确定"
                    ))
                    
        open_dialog = ft.FilePicker(on_result=on_open_dialog)
        self.page.overlay.append(open_dialog)
        self.page.update()
        open_dialog.pick_files(
            dialog_title="导入设置",
            allowed_extensions=["json"],
            allow_multiple=False
        )
        
    def _build_ui(self):
        """构建UI"""
        return ft.Column([
            ft.Text("设置管理", size=24, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            self._build_settings_view(),
            ft.Divider(),
            self._build_actions(),
        ], spacing=20, expand=True)
        
    def main(self, page: ft.Page):
        """主函数"""
        self.page = page
        page.title = "设置管理器演示"
        page.theme_mode = ft.ThemeMode.LIGHT if get_setting("ui", "theme") == "light" else ft.ThemeMode.DARK
        
        # 观察主题变化
        def theme_observer(section, key, value):
            if section == "ui" and key == "theme":
                page.theme_mode = ft.ThemeMode.LIGHT if value == "light" else ft.ThemeMode.DARK
                page.update()
                
        settings.add_observer(theme_observer)
        
        page.add(self._build_ui())

# 如果直接运行此文件，则启动演示
if __name__ == "__main__":
    ft.app(SettingsPreview().main, view=ft.WEB_BROWSER, port=8080)