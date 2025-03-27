"""
模型面板模块

负责:
- 管理AI模型资源
- 处理模型的导入和加载
- 显示模型状态和信息
- 提供模型配置界面

模型以卡片形式展示,显示状态标签和类型图标
"""

from typing import Dict, Optional, Callable
import flet as ft
from ...config.language import get_text
from ...entity.model_repository import Model, ModelType, ModelStatus

class ModelCard(ft.UserControl):
    """模型卡片"""
    
    def __init__(
        self,
        model: Model,
        on_select: Optional[Callable[[str], None]] = None,
        on_config: Optional[Callable[[str], None]] = None
    ):
        super().__init__()
        self.model = model
        self.on_select = on_select
        self.on_config = on_config
        
    def build(self):
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        # 标题栏
                        ft.Row(
                            [
                                ft.Icon(
                                    name=self._get_type_icon(),
                                    color=self._get_type_color()
                                ),
                                ft.Text(
                                    self.model.name,
                                    size=16,
                                    weight=ft.FontWeight.BOLD
                                ),
                                ft.Container(
                                    content=ft.Text(
                                        self._get_status_text(),
                                        size=12,
                                        color=ft.colors.WHITE
                                    ),
                                    bgcolor=self._get_status_color(),
                                    padding=5,
                                    border_radius=3
                                )
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                        ),
                        
                        # 描述
                        ft.Text(
                            self.model.description,
                            size=14,
                            color=ft.colors.BLACK54
                        ),
                        
                        ft.Divider(),
                        
                        # 操作按钮
                        ft.Row(
                            [
                                ft.TextButton(
                                    text=get_text("model.select"),
                                    icon=ft.icons.CHECK,
                                    on_click=lambda _: (
                                        self.on_select and 
                                        self.on_select(self.model.id)
                                    )
                                ),
                                ft.TextButton(
                                    text=get_text("model.config"),
                                    icon=ft.icons.SETTINGS,
                                    on_click=lambda _: (
                                        self.on_config and 
                                        self.on_config(self.model.id)
                                    )
                                )
                            ],
                            alignment=ft.MainAxisAlignment.END
                        )
                    ],
                    spacing=10
                ),
                padding=10
            )
        )
        
    def _get_type_icon(self) -> str:
        """获取类型图标"""
        icons = {
            ModelType.LLM: ft.icons.CHAT,
            ModelType.IMAGE: ft.icons.IMAGE,
            ModelType.AUDIO: ft.icons.AUDIOTRACK,
            ModelType.VIDEO: ft.icons.VIDEOCAM,
            ModelType.MULTIMODAL: ft.icons.APPS
        }
        return icons.get(self.model.type, ft.icons.HELP)
        
    def _get_type_color(self) -> str:
        """获取类型颜色"""
        colors = {
            ModelType.LLM: "#2196F3",
            ModelType.IMAGE: "#4CAF50",
            ModelType.AUDIO: "#FF9800",
            ModelType.VIDEO: "#F44336",
            ModelType.MULTIMODAL: "#9C27B0"
        }
        return colors.get(self.model.type, "#666666")
        
    def _get_status_text(self) -> str:
        """获取状态文本"""
        return get_text(f"model.status.{self.model.status.value}")
        
    def _get_status_color(self) -> str:
        """获取状态颜色"""
        colors = {
            ModelStatus.AVAILABLE: "#4CAF50",
            ModelStatus.UNAVAILABLE: "#F44336",
            ModelStatus.LOADING: "#FF9800",
            ModelStatus.ERROR: "#F44336"
        }
        return colors.get(self.model.status, "#666666")

class ModelPanel(ft.UserControl):
    """模型面板"""
    
    def __init__(
        self,
        on_model_select: Optional[Callable[[str], None]] = None,
        on_model_config: Optional[Callable[[str], None]] = None,
        on_import: Optional[Callable[[], None]] = None
    ):
        super().__init__()
        self.on_model_select = on_model_select
        self.on_model_config = on_model_config
        self.on_import = on_import
        self._models: Dict[str, Model] = {}
        self._search_text = ""
        
    def build(self):
        return ft.Column(
            [
                # 工具栏
                ft.Row(
                    [
                        ft.TextField(
                            prefix_icon=ft.icons.SEARCH,
                            hint_text=get_text("model.search"),
                            expand=True,
                            on_change=self._on_search_change
                        ),
                        ft.IconButton(
                            icon=ft.icons.ADD,
                            tooltip=get_text("model.import"),
                            on_click=lambda _: (
                                self.on_import and self.on_import()
                            )
                        )
                    ]
                ),
                
                # 模型列表
                ft.Column(
                    [
                        ModelCard(
                            model=model,
                            on_select=self.on_model_select,
                            on_config=self.on_model_config
                        )
                        for model in self._filter_models()
                    ],
                    spacing=10,
                    scroll=ft.ScrollMode.AUTO
                )
            ],
            spacing=20
        )
        
    def _on_search_change(self, e):
        """搜索文本变化"""
        self._search_text = e.control.value.lower()
        self.update()
        
    def _filter_models(self) -> List[Model]:
        """过滤模型列表"""
        return [
            model for model in self._models.values()
            if not self._search_text or
            self._search_text in model.name.lower() or
            self._search_text in model.description.lower()
        ]
        
    def update_models(self, models: Dict[str, Model]):
        """更新模型列表"""
        self._models = models
        self.update() 