"""
模型仓库模块

负责:
- 管理AI模型资源
- 处理模型的版本控制
- 提供模型的元数据管理
- 支持模型的导入导出

以表格或卡片形式展示,支持筛选和排序
"""

from typing import Dict, List, Optional
from enum import Enum
import json
import os
from pathlib import Path
from datetime import datetime

class ModelType(Enum):
    """模型类型"""
    LLM = "llm"              # 大语言模型
    IMAGE = "image"          # 图像模型
    AUDIO = "audio"          # 音频模型
    VIDEO = "video"          # 视频模型
    MULTIMODAL = "multimodal"# 多模态模型

class ModelProvider(Enum):
    """模型提供商"""
    LOCAL = "local"          # 本地模型
    OPENAI = "openai"        # OpenAI
    ANTHROPIC = "anthropic"  # Anthropic
    HUGGINGFACE = "huggingface"  # HuggingFace
    CUSTOM = "custom"        # 自定义

class ModelStatus(Enum):
    """模型状态"""
    AVAILABLE = "available"    # 可用
    UNAVAILABLE = "unavailable"# 不可用
    LOADING = "loading"        # 加载中
    ERROR = "error"           # 错误

class Model:
    """模型定义类"""
    
    def __init__(
        self,
        id: str,
        name: str,
        type: ModelType,
        provider: ModelProvider,
        version: str,
        description: str = None,
        config: Dict = None,
        status: ModelStatus = ModelStatus.UNAVAILABLE
    ):
        self.id = id
        self.name = name
        self.type = type
        self.provider = provider
        self.version = version
        self.description = description or ""
        self.config = config or {}
        self.status = status
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "provider": self.provider.value,
            "version": self.version,
            "description": self.description,
            "config": self.config,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Model':
        """从字典创建实例"""
        model = cls(
            id=data["id"],
            name=data["name"],
            type=ModelType(data["type"]),
            provider=ModelProvider(data["provider"]),
            version=data["version"],
            description=data.get("description"),
            config=data.get("config", {}),
            status=ModelStatus(data.get("status", "unavailable"))
        )
        model.created_at = data.get("created_at", model.created_at)
        model.updated_at = data.get("updated_at", model.updated_at)
        return model

class ModelRepository:
    """模型仓库类"""
    
    def __init__(self):
        self._models: Dict[str, Model] = {}
        self._load_models()
        
    def _load_models(self):
        """加载模型配置"""
        models_path = Path(os.path.dirname(__file__)) / "../config/models.json"
        
        if models_path.exists():
            try:
                with open(models_path, "r", encoding="utf-8") as f:
                    models_data = json.load(f)
                    for model_data in models_data:
                        model = Model.from_dict(model_data)
                        self._models[model.id] = model
            except Exception as e:
                print(f"加载模型配置失败: {e}")
                
    def _save_models(self):
        """保存模型配置"""
        models_path = Path(os.path.dirname(__file__)) / "../config/models.json"
        
        try:
            models_data = [model.to_dict() for model in self._models.values()]
            with open(models_path, "w", encoding="utf-8") as f:
                json.dump(models_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"保存模型配置失败: {e}")
            
    def get_model(self, model_id: str) -> Optional[Model]:
        """获取模型"""
        return self._models.get(model_id)
        
    def get_models_by_type(self, type: ModelType) -> List[Model]:
        """获取指定类型的所有模型"""
        return [
            model for model in self._models.values()
            if model.type == type
        ]
        
    def get_models_by_provider(self, provider: ModelProvider) -> List[Model]:
        """获取指定提供商的所有模型"""
        return [
            model for model in self._models.values()
            if model.provider == provider
        ]
        
    def add_model(self, model: Model):
        """添加模型"""
        if model.id in self._models:
            raise ValueError(f"模型 {model.id} 已存在")
        self._models[model.id] = model
        self._save_models()
        
    def update_model(self, model_id: str, **kwargs):
        """更新模型"""
        if model_id not in self._models:
            raise ValueError(f"模型 {model_id} 不存在")
            
        model = self._models[model_id]
        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)
        
        model.updated_at = datetime.now().isoformat()
        self._save_models()
        
    def remove_model(self, model_id: str):
        """删除模型"""
        if model_id in self._models:
            del self._models[model_id]
            self._save_models()
            
    def update_model_status(self, model_id: str, status: ModelStatus):
        """更新模型状态"""
        if model_id in self._models:
            self._models[model_id].status = status
            self._save_models()

# 创建全局模型仓库实例
model_repository = ModelRepository() 