"""
素材仓库模块

负责:
- 管理素材资源
- 处理素材的分类和标签
- 提供素材的预览功能
- 支持素材的导入导出

以网格或列表形式展示,支持预览和快速使用
"""

from typing import Dict, List, Optional, Set
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

class Material:
    """素材定义类"""
    
    def __init__(
        self,
        id: str,
        name: str,
        material_type: MaterialType,
        path: str,
        description: str = None,
        tags: Set[str] = None,
        metadata: Dict = None
    ):
        self.id = id
        self.name = name
        self.type = material_type
        self.path = path
        self.description = description or ""
        self.tags = tags or set()
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
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
        
        if materials_path.exists():
            try:
                with open(materials_path, "r", encoding="utf-8") as f:
                    materials_data = json.load(f)
                    for material_data in materials_data:
                        material = Material.from_dict(material_data)
                        self._materials[material.id] = material
                        self._tags.update(material.tags)
            except Exception as e:
                print(f"加载素材配置失败: {e}")
        else:
            self.add_material(Material(id="default", name="默认素材", material_type=MaterialType.TEXT, path=Path(os.path.dirname(__file__)) / "../assets/materials/default.txt"))
                
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
