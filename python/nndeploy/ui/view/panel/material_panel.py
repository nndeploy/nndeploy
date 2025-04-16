"""
素材面板模块

负责:
- 管理素材资源
- 处理素材的导入和预览
- 支持素材拖放到画布
- 提供素材分类管理
- 在左侧边栏展示

素材支持缩略图预览,可切换列表和网格视图
"""

"""
素材仓库模块

负责:
- 管理素材资源
- 处理素材的分类和标签
- 提供素材的预览功能
- 支持素材的导入导出

以网格或列表形式展示,支持预览和快速使用
"""

from typing import Dict, List, Optional, Set, Union
from enum import Enum
import json
import os
from pathlib import Path
from datetime import datetime
import shutil

class MaterialType(Enum):
    """素材类型"""
    TEXT = "text"        # 文本
    IMAGE = "image"      # 图片
    AUDIO = "audio"      # 音频
    VIDEO = "video"      # 视频
    FILE = "file"        # 文件
    
    @classmethod
    def from_str(cls, type_str: str):
        """从字符串获取枚举值"""
        try:
            return cls(type_str)
        except ValueError:
            return cls.FILE  # 默认返回文件类型

class Material:
    """素材定义类"""
    
    def __init__(
        self,
        id: str,
        name: str,
        material_type: Union[MaterialType, str],
        path: str,
        thumbnail: str = None,
        description: str = None,
        tags: Set[str] = None,
        metadata: Dict = None
    ):
        # id: 素材的唯一标识符，用于在系统中唯一引用该素材
        # 不同于name，id通常是系统生成的，不会随用户操作而改变
        self.id = id
        
        # name: 素材的显示名称，用于在UI界面上展示，可由用户自定义
        # 与id不同，name可以重复，主要用于用户识别和搜索
        self.name = name
        
        # 素材类型（文本、图片、音频等）
        if isinstance(material_type, str):
            m_type = MaterialType.from_str(material_type)
        else:
            m_type = material_type
        self.type = m_type
        
        # 素材文件的相对路径
        self.path = path
        
        # 素材的详细描述信息
        self.description = description or ""
        
        # tags: 素材的标签集合，用于分类和筛选
        # 与id和name不同，tags是多值属性，一个素材可以有多个标签
        # 标签用于对素材进行分组和快速查找
        self.tags = tags or set()
        
        # 存储素材的额外元数据
        self.metadata = metadata or {}
        
        # 记录素材创建时间
        self.created_at = datetime.now().isoformat()
        
        # 记录素材最后更新时间，初始与创建时间相同
        self.updated_at = self.created_at
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "path": self.path,
            "description": self.description,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Material':
        """从字典创建实例"""
        material = cls(
            id=data["id"],
            name=data["name"],
            type=MaterialType(data["type"]),
            path=data["path"],
            description=data.get("description"),
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {})
        )
        material.created_at = data.get("created_at", material.created_at)
        material.updated_at = data.get("updated_at", material.updated_at)
        return material

class MaterialRepository:
    """素材仓库类"""
    
    def __init__(self, materials_dir: Path = None):
        self._materials: Dict[str, Material] = {}
        self._tags: Set[str] = set()
        self._materials_dir = materials_dir or Path(os.path.dirname(__file__)) / "../assets/materials"
        self._materials_dir.mkdir(parents=True, exist_ok=True)
        self._load_materials()
        
    def _load_materials(self):
        """加载素材配置"""
        materials_path = Path(os.path.dirname(__file__)) / "../assets/materials.json"
        
        # if materials_path.exists():
        #     print(f"加载素材配置: {materials_path}")
        #     try:
        #         with open(materials_path, "r", encoding="utf-8") as f:
        #             materials_data = json.load(f)
        #             for material_data in materials_data:
        #                 material = Material.from_dict(material_data)
        #                 self._materials[material.id] = material
        #                 self._tags.update(material.tags)
        #     except Exception as e:
        #         print(f"加载素材配置失败: {e}")
        # else:
        #     self.add_material(Material(id="default", name="默认素材", material_type=MaterialType.TEXT, path=Path(os.path.dirname(__file__)) / "../assets/materials/default.txt"))
                
    def _save_materials(self):
        """保存素材配置"""
        materials_path = Path(os.path.dirname(__file__)) / "../assets/materials.json"
        if not materials_path.exists():
            materials_path.parent.mkdir(parents=True, exist_ok=True)
            materials_path.touch()
        try:
            materials_data = [
                material.to_dict() for material in self._materials.values()
            ]
            with open(materials_path, "w", encoding="utf-8") as f:
                json.dump(materials_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"保存素材配置失败: {e}")
            
    def get_material(self, material_id: str) -> Optional[Material]:
        """获取素材"""
        return self._materials.get(material_id)
        
    def get_materials_by_type(self, material_type: MaterialType) -> List[Material]:
        """获取指定类型的所有素材"""
        return [
            material for material in self._materials.values()
            if material.type == material_type
        ]
        
    def get_materials_by_tag(self, tag: str) -> List[Material]:
        """获取指定标签的所有素材"""
        return [
            material for material in self._materials.values()
            if tag in material.tags
        ]
        
    def get_all_tags(self) -> Set[str]:
        """获取所有标签"""
        return self._tags.copy()
        
    def add_material(self, material: Material, file_path: Optional[str] = None):
        """添加素材
        
        Args:
            material: 素材对象
            file_path: 源文件路径(如果需要导入文件)
        """
        if material.id in self._materials:
            raise ValueError(f"素材 {material.id} 已存在")
            
        # 如果提供了文件路径,复制文件到素材目录
        if file_path:
            target_path = self._materials_dir / f"{material.id}{Path(file_path).suffix}"
            shutil.copy2(file_path, target_path)
            material.path = str(target_path.relative_to(self._materials_dir))
            
        self._materials[material.id] = material
        self._tags.update(material.tags)
        self._save_materials()
        
    def update_material(self, material_id: str, **kwargs):
        """更新素材"""
        if material_id not in self._materials:
            raise ValueError(f"素材 {material_id} 不存在")
            
        material = self._materials[material_id]
        
        # 更新标签
        if "tags" in kwargs:
            self._tags.difference_update(material.tags)
            material.tags = set(kwargs["tags"])
            self._tags.update(material.tags)
            del kwargs["tags"]
            
        # 更新其他属性
        for key, value in kwargs.items():
            if hasattr(material, key):
                setattr(material, key, value)
                
        material.updated_at = datetime.now().isoformat()
        self._save_materials()
        
    def remove_material(self, material_id: str):
        """删除素材"""
        if material_id in self._materials:
            material = self._materials[material_id]
            
            # 删除文件
            material_path = self._materials_dir / material.path
            if material_path.exists():
                material_path.unlink()
                
            # 更新标签集合
            self._tags.difference_update(material.tags)
            
            # 删除素材记录
            del self._materials[material_id]
            self._save_materials()

# 创建全局素材仓库实例
material_repository = MaterialRepository() 


"""
素材仓库测试代码

用于测试素材仓库的功能，包括：
- 素材的添加、更新、删除
- 素材的查询和过滤
- 标签管理
- 文件操作
"""

def test_material_repository():
    """测试素材仓库的基本功能"""
    # 测试初始化
    repo = MaterialRepository()
    
    # 测试添加素材
    import tempfile
    import os
    
    # 创建临时文件作为测试素材
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
        temp_path = temp.name
        
    try:
        # 创建测试素材
        material = Material(
            id="test-material-1",
            name="测试素材1",
            material_type=MaterialType.TEXT,
            tags={"测试", "文本"},
            description="这是一个测试素材",
            path=Path(os.path.dirname(__file__)) / "../assets/materials/default.txt"
        )
        
        # 测试添加素材
        repo.add_material(material, temp_path)
        # assert "test-material-1" in repo.get_material()
        assert "测试" in repo.get_all_tags()
        assert "文本" in repo.get_all_tags()
        
        # 测试更新素材
        repo.update_material("test-material-1", 
                            name="更新后的素材", 
                            tags={"测试", "更新"},
                            description="这是更新后的描述")
        
        updated_material = repo.get_material("test-material-1")
        assert updated_material.name == "更新后的素材"
        assert updated_material.tags == {"测试", "更新"}
        assert updated_material.description == "这是更新后的描述"
        assert "图片" not in repo.get_all_tags()
        assert "更新" in repo.get_all_tags()
        
        # 测试按标签过滤
        filtered_materials = repo.get_materials_by_tag("测试")
        assert len(filtered_materials) == 1
        assert filtered_materials[0].id == "test-material-1"
        
        # 测试按类型过滤
        filtered_materials = repo.get_materials_by_type(MaterialType.TEXT)
        print(len(filtered_materials))
        
        # 测试删除素材
        repo.remove_material("test-material-1")
        
        # 测试标签清理
        assert "测试" not in repo.get_all_tags()
        assert "更新" not in repo.get_all_tags()
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
    print("素材仓库测试完成")

if __name__ == "__main__":
    test_material_repository()


from typing import Dict, List, Optional, Callable
import flet as ft
from nndeploy.ui.config.language import get_text
from nndeploy.ui.entity.material_repository import Material, MaterialType

class MaterialCard(ft.Container):
    """素材卡片"""
    
    def __init__(
        self,
        material: Material,
        on_select: Optional[Callable[[str], None]] = None,
        on_drag_start: Optional[Callable[[str, float, float], None]] = None
    ):
        self.material = material
        self.on_select = on_select
        self.on_drag_start = on_drag_start
        
        # 构建卡片内容
        card_content = ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        # 预览图
                        ft.Container(
                            content=self._build_preview(),
                            height=120,
                            border_radius=ft.border_radius.vertical(5, 0)
                        ),
                        
                        # 信息
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text(
                                        self.material.name,
                                        size=14,
                                        weight=ft.FontWeight.BOLD
                                    ),
                                    ft.Text(
                                        self.material.description,
                                        size=12,
                                        color=ft.colors.BLACK54,
                                        max_lines=2,
                                        overflow=ft.TextOverflow.ELLIPSIS
                                    )
                                ],
                                spacing=5
                            ),
                            padding=10
                        )
                    ],
                    spacing=0
                )
            )
        )
        
        # 使用手势检测器包装卡片
        gesture_detector = ft.GestureDetector(
            content=card_content,
            mouse_cursor=ft.MouseCursor.MOVE,
            on_tap=lambda _: (
                self.on_select and self.on_select(self.material.id)
            ),
            on_pan_start=self._on_drag_start
        )
        
        # 初始化父类
        super().__init__(content=gesture_detector)
        
    def _build_preview(self) -> ft.Control:
        """构建预览控件"""
        if self.material.type == "image":
            return ft.Image(
                src=self.material.path,
                fit=ft.ImageFit.COVER
            )
        elif self.material.type == "video":
            return ft.Stack(
                [
                    ft.Image(
                        src=self.material.thumbnail,
                        fit=ft.ImageFit.COVER
                    ),
                    ft.Container(
                        content=ft.Icon(
                            ft.icons.PLAY_CIRCLE,
                            size=40,
                            color=ft.colors.WHITE
                        ),
                        alignment=ft.alignment.center
                    )
                ]
            )
        else:
            return ft.Icon(
                self._get_type_icon(),
                size=40,
                color=ft.colors.BLACK54
            )
            
    def _get_type_icon(self) -> str:
        """获取类型图标"""
        icons = {
            "text": ft.icons.TEXT_FIELDS,
            "audio": ft.icons.AUDIOTRACK,
            "file": ft.icons.INSERT_DRIVE_FILE
        }
        return icons.get(self.material.type, ft.icons.HELP)
        
    def _on_drag_start(self, e: ft.DragStartEvent):
        """开始拖动"""
        if self.on_drag_start:
            self.on_drag_start(self.material.id, e.local_x, e.local_y)

class MaterialPanel(ft.Column):
    """素材面板"""
    
    def __init__(
        self,
        on_material_select: Optional[Callable[[str], None]] = None,
        on_material_drag_start: Optional[Callable[[str, float, float], None]] = None,
        on_import: Optional[Callable[[], None]] = None
    ):
        self.on_material_select = on_material_select
        self.on_material_drag_start = on_material_drag_start
        self.on_import = on_import
        self._materials: Dict[str, Material] = {}
        self._search_text = ""
        self._grid_view = True
        
        # 工具栏
        self.toolbar = ft.Row(
            [
                ft.TextField(
                    prefix_icon=ft.icons.SEARCH,
                    hint_text=get_text("material.search"),
                    expand=True,
                    on_change=self._on_search_change
                ),
                ft.IconButton(
                    icon=ft.icons.GRID_VIEW if self._grid_view
                    else ft.icons.LIST,
                    tooltip=get_text(
                        "material.gridView" if self._grid_view
                        else "material.listView"
                    ),
                    on_click=self._toggle_view
                ),
                ft.IconButton(
                    icon=ft.icons.ADD,
                    tooltip=get_text("material.import"),
                    on_click=lambda _: (
                        self.on_import and self.on_import()
                    )
                )
            ]
        )
        
        # 素材列表
        self.material_grid = ft.GridView(
            runs_count=2 if self._grid_view else 1,
            max_extent=200 if self._grid_view else 400,
            child_aspect_ratio=1.0 if self._grid_view else 3.0,
            spacing=10,
            run_spacing=10
        )
        
        # 初始化父类
        super().__init__(
            [self.toolbar, self.material_grid],
            spacing=20
        )
        
    def _on_search_change(self, e):
        """搜索文本变化"""
        self._search_text = e.control.value.lower()
        self._update_material_grid()
        self.update()
        
    def _toggle_view(self, _):
        """切换视图模式"""
        self._grid_view = not self._grid_view
        self.material_grid.runs_count = 2 if self._grid_view else 1
        self.material_grid.max_extent = 200 if self._grid_view else 400
        self.material_grid.child_aspect_ratio = 1.0 if self._grid_view else 3.0
        self._update_material_grid()
        self.update()
        
    def _filter_materials(self) -> List[Material]:
        """过滤素材列表"""
        return [
            material for material in self._materials.values()
            if not self._search_text or
            self._search_text in material.name.lower() or
            self._search_text in material.description.lower()
        ]
    
    def _update_material_grid(self):
        """更新素材网格"""
        self.material_grid.controls = [
            MaterialCard(
                material=material,
                on_select=self.on_material_select,
                on_drag_start=self.on_material_drag_start
            )
            for material in self._filter_materials()
        ]
        
    def update_materials(self, materials: Dict[str, Material]):
        """更新素材列表"""
        self._materials = materials
        self._update_material_grid()
        self.update()

def main(page: ft.Page):
    """预览演示"""
    page.title = "素材面板演示"
    
    # 创建测试素材
    materials = {
        "1": Material(
            id="1",
            name="测试图片",
            description="这是一个测试图片素材",
            material_type="image",
            path="https://picsum.photos/200/300",
            thumbnail="https://picsum.photos/200/300"
        ),
        "2": Material(
            id="2",
            name="测试视频",
            description="这是一个测试视频素材，描述比较长以测试多行显示效果",
            material_type="video",
            path="video.mp4",
            thumbnail="https://picsum.photos/200/300"
        ),
        "3": Material(
            id="3",
            name="测试文本",
            description="这是一个测试文本素材",
            material_type="text",
            path="text.txt"
        ),
        "4": Material(
            id="4",
            name="测试音频",
            description="这是一个测试音频素材",
            material_type="audio",
            path="audio.mp3"
        )
    }
    
    def on_material_select(material_id: str):
        page.snack_bar = ft.SnackBar(
            content=ft.Text(f"选择了素材: {materials[material_id].name}")
        )
        page.snack_bar.open = True
        page.update()
    
    def on_material_drag_start(material_id: str, x: float, y: float):
        page.snack_bar = ft.SnackBar(
            content=ft.Text(f"开始拖动素材: {materials[material_id].name} 位置: ({x}, {y})")
        )
        page.snack_bar.open = True
        page.update()
    
    def on_import():
        page.snack_bar = ft.SnackBar(
            content=ft.Text("导入素材")
        )
        page.snack_bar.open = True
        page.update()
    
    # 创建素材面板
    panel = MaterialPanel(
        on_material_select=on_material_select,
        on_material_drag_start=on_material_drag_start,
        on_import=on_import
    )
    
    # 先将面板添加到页面，然后再更新素材
    page.add(panel)
    panel.update_materials(materials)

if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER, port=8080)
