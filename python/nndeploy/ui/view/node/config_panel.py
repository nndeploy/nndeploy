"""
配置面板模块

负责:
- 显示节点的配置选项
- 处理配置的实时更新
- 管理配置的验证
- 提供配置的预设模板

采用选项卡式设计,配置实时生效
"""

from typing import Dict, Any, Optional, Callable
import flet as ft
from ...config.language import get_text

class ConfigField(ft.UserControl):
    """配置字段基类"""
    
    def __init__(
        self,
        key: str,
        label: str,
        value: Any,
        on_change: Optional[Callable[[str, Any], None]] = None
    ):
        super().__init__()
        self.key = key
        self.label = label
        self.value = value
        self.on_change = on_change
        
    def _notify_change(self, value: Any):
        """通知值变化"""
        self.value = value
        if self.on_change:
            self.on_change(self.key, value)

class StringField(ConfigField):
    """字符串字段"""
    
    def build(self):
        return ft.TextField(
            label=self.label,
            value=str(self.value),
            on_change=lambda e: self._notify_change(e.control.value)
        )

class NumberField(ConfigField):
    """数值字段"""
    
    def __init__(
        self,
        key: str,
        label: str,
        value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: float = 1,
        on_change: Optional[Callable[[str, float], None]] = None
    ):
        super().__init__(key, label, value, on_change)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        
    def build(self):
        return ft.Row(
            [
                ft.Text(self.label),
                ft.Slider(
                    min=self.min_value if self.min_value is not None else 0,
                    max=self.max_value if self.max_value is not None else 100,
                    value=float(self.value),
                    divisions=int((self.max_value - self.min_value) / self.step)
                    if self.max_value is not None and self.min_value is not None
                    else None,
                    label="{value}",
                    on_change=lambda e: self._notify_change(e.control.value)
                )
            ]
        )

class BooleanField(ConfigField):
    """布尔字段"""
    
    def build(self):
        return ft.Checkbox(
            label=self.label,
            value=bool(self.value),
            on_change=lambda e: self._notify_change(e.control.value)
        )

class SelectField(ConfigField):
    """选择字段"""
    
    def __init__(
        self,
        key: str,
        label: str,
        value: Any,
        options: Dict[str, Any],
        on_change: Optional[Callable[[str, Any], None]] = None
    ):
        super().__init__(key, label, value, on_change)
        self.options = options
        
    def build(self):
        return ft.Dropdown(
            label=self.label,
            value=str(self.value),
            options=[
                ft.dropdown.Option(key=str(k), text=str(v))
                for k, v in self.options.items()
            ],
            on_change=lambda e: self._notify_change(
                self.options[e.control.value]
            )
        )

class ConfigPanel(ft.UserControl):
    """节点配置面板"""
    
    def __init__(
        self,
        title: str = "",
        fields: Dict[str, Dict] = None,
        values: Dict[str, Any] = None,
        on_change: Optional[Callable[[str, Any], None]] = None
    ):
        super().__init__()
        self.title = title
        self._fields = fields or {}
        self._values = values or {}
        self.on_change = on_change
        
    def build(self):
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        # 标题
                        ft.Text(
                            self.title,
                            size=20,
                            weight=ft.FontWeight.BOLD
                        ),
                        
                        # 分隔线
                        ft.Divider(),
                        
                        # 配置字段
                        ft.Column(
                            [
                                self._build_field(key, field)
                                for key, field in self._fields.items()
                            ],
                            scroll=ft.ScrollMode.AUTO,
                            spacing=10
                        )
                    ],
                    spacing=20
                ),
                padding=20
            )
        )
        
    def _build_field(self, key: str, field: Dict) -> ft.Control:
        """构建配置字段"""
        field_type = field.get("type", "string")
        value = self._values.get(key, field.get("default"))
        
        if field_type == "string":
            return StringField(
                key=key,
                label=field.get("label", key),
                value=value,
                on_change=self.on_change
            )
            
        elif field_type == "number":
            return NumberField(
                key=key,
                label=field.get("label", key),
                value=value,
                min_value=field.get("min"),
                max_value=field.get("max"),
                step=field.get("step", 1),
                on_change=self.on_change
            )
            
        elif field_type == "boolean":
            return BooleanField(
                key=key,
                label=field.get("label", key),
                value=value,
                on_change=self.on_change
            )
            
        elif field_type == "select":
            return SelectField(
                key=key,
                label=field.get("label", key),
                value=value,
                options=field.get("options", {}),
                on_change=self.on_change
            )
            
        else:
            return ft.Text(f"Unsupported field type: {field_type}")
            
    def update_values(self, values: Dict[str, Any]):
        """更新配置值"""
        self._values.update(values)
        self.update() 