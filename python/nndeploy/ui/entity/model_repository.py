"""
模型仓库模块

负责:
- 管理AI模型资源
- 处理模型的版本控制
- 提供模型的元数据管理
- 支持模型的导入导出

以表格或卡片形式展示,支持筛选和排序
"""

from typing import Dict, List, Optional, Set
from enum import Enum
import json
import os
from pathlib import Path
from datetime import datetime

class ModelType(Enum):
    """模型类型"""
    LLM = "llm"              # 大语言模型
    CV = "cv"          # 图像模型
    MULTIMODAL = "multimodal"# 多模态模型

class ModelProvider(Enum):
    """模型提供商"""
    LOCAL = "local"          # 本地模型
    
    OPENAI = "openai"        # OpenAI
    ANTHROPIC = "anthropic"  # Anthropic
    HUGGINGFACE = "huggingface"  # HuggingFace
    GEMINI = "gemini"        # Gemini
    QIANFAN = "qianfan"      # 千帆
    AZURE = "azure"          # Azure
    COZE = "coze"            # Coze
    DEEPSEEK = "deepseek"    # DeepSeek
    QIANWEN = "qianwen"      # 千问
    XUNFEI = "xunfei"        # 讯飞
    STABILITY = "stability"  # Stability AI
    MIDJOURNEY = "midjourney"# Midjourney
    DALLE = "dalle"          # DALL-E
    LEONARDO = "leonardo"    # Leonardo.AI
    COMFYUI = "comfyui"      # ComfyUI
    SDWEBUI = "sdwebui"      # Stable Diffusion WebUI
    PIXART = "pixart"        # PixArt-α
    WANX = "wanx"            # 腾讯万象
    ZHIPUAI = "zhipuai"      # 智谱AI
    BAIDU = "baidu"          # 百度文心一格
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
        status: ModelStatus = ModelStatus.UNAVAILABLE,
        tags: Set[str] = None,
        metadata: Dict = None
    ):
        # id: 模型的唯一标识符，用于在系统中唯一引用该模型
        # 不同于name，id通常是系统生成的，不会随用户操作而改变
        self.id = id
        
        # name: 模型的显示名称，用于在UI界面上展示，可由用户自定义
        # 与id不同，name可以重复，主要用于用户识别和搜索
        self.name = name
        
        # 模型类型（文本生成、图像生成等）
        self.type = type
        
        # 模型提供商（OpenAI、本地等）
        self.provider = provider
        
        # 模型版本号，用于区分同一模型的不同版本
        self.version = version
        
        # 模型的详细描述信息
        self.description = description or ""
        
        # 存储模型的配置信息，如API密钥、服务器地址等
        self.config = config or {}
        
        # 模型当前状态（可用、不可用、加载中、错误）
        self.status = status
        
        # tags: 模型的标签集合，用于分类和筛选
        # 一个模型可以有多个标签，用于对模型进行分组和快速查找
        self.tags = tags or set()
        
        # 存储模型的额外元数据，如模型大小、参数数量等
        self.metadata = metadata or {}
        
        # 记录模型创建时间
        self.created_at = datetime.now().isoformat()
        
        # 记录模型最后更新时间，初始与创建时间相同
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
            status=ModelStatus(data.get("status", "unavailable")),
            tags=data.get("tags", set()),
            metadata=data.get("metadata", {})
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
        models_path = Path(os.path.dirname(__file__)) / "../assets/models.json"
        
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
        models_path = Path(os.path.dirname(__file__)) / "../assets/models.json"
        
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
        
    def add_model_from_file(self, file_path: str):
        """从文件json文件添加模型"""
        with open(file_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
            # 处理单个模型或多个模型的情况
            if isinstance(model_data, list):
                # 多个模型的情况
                for single_model_data in model_data:
                    model = Model.from_dict(single_model_data)
                    self.add_model(model)
            else:
                # 单个模型的情况
                model = Model.from_dict(model_data)
                self.add_model(model)
        
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