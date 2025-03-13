"""
JSON工具模块

负责:
- 处理JSON数据的序列化
- 提供JSON格式化功能
- 处理JSON数据验证
- 支持JSON数据转换

JSON数据格式化显示,支持折叠展开
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import jsonschema

class JsonUtils:
    """JSON工具类"""
    
    @staticmethod
    def load(file_path: str) -> Any:
        """从文件加载JSON数据"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    @staticmethod
    def save(data: Any, file_path: str, indent: int = 4):
        """保存JSON数据到文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            
    @staticmethod
    def format_str(
        data: Any,
        indent: int = 4,
        sort_keys: bool = False
    ) -> str:
        """格式化JSON字符串"""
        return json.dumps(
            data,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=False
        )
        
    @staticmethod
    def parse_str(json_str: str) -> Any:
        """解析JSON字符串"""
        return json.loads(json_str)
        
    @staticmethod
    def validate(data: Any, schema: Dict):
        """验证JSON数据
        
        Args:
            data: JSON数据
            schema: JSON Schema
            
        Raises:
            jsonschema.exceptions.ValidationError: 验证失败
        """
        jsonschema.validate(instance=data, schema=schema)
        
    @staticmethod
    def is_valid(data: Any, schema: Dict) -> bool:
        """检查JSON数据是否有效"""
        try:
            JsonUtils.validate(data, schema)
            return True
        except jsonschema.exceptions.ValidationError:
            return False
            
    @staticmethod
    def merge(base: Dict, update: Dict, overwrite: bool = True) -> Dict:
        """合并JSON对象
        
        Args:
            base: 基础对象
            update: 更新对象
            overwrite: 是否覆盖已存在的键
            
        Returns:
            合并后的对象
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) \
                    and isinstance(value, dict):
                result[key] = JsonUtils.merge(
                    result[key],
                    value,
                    overwrite
                )
            elif key not in result or overwrite:
                result[key] = value
                
        return result
        
    @staticmethod
    def diff(old: Dict, new: Dict) -> Dict:
        """比较两个JSON对象的差异
        
        Returns:
            包含added/removed/modified键的差异对象
        """
        result = {
            "added": {},
            "removed": {},
            "modified": {}
        }
        
        # 检查新增和修改的键
        for key, value in new.items():
            if key not in old:
                result["added"][key] = value
            elif old[key] != value:
                result["modified"][key] = {
                    "old": old[key],
                    "new": value
                }
                
        # 检查删除的键
        for key, value in old.items():
            if key not in new:
                result["removed"][key] = value
                
        return result
        
    @staticmethod
    def flatten(
        data: Dict,
        separator: str = ".",
        prefix: str = ""
    ) -> Dict:
        """扁平化JSON对象
        
        Args:
            data: JSON对象
            separator: 键分隔符
            prefix: 键前缀
            
        Returns:
            扁平化后的对象
        """
        result = {}
        
        for key, value in data.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            
            if isinstance(value, dict):
                result.update(
                    JsonUtils.flatten(value, separator, new_key)
                )
            else:
                result[new_key] = value
                
        return result
        
    @staticmethod
    def unflatten(
        data: Dict,
        separator: str = "."
    ) -> Dict:
        """还原扁平化的JSON对象"""
        result = {}
        
        for key, value in data.items():
            parts = key.split(separator)
            target = result
            
            for part in parts[:-1]:
                target = target.setdefault(part, {})
                
            target[parts[-1]] = value
            
        return result
        
    @staticmethod
    def to_json_string(obj: Any) -> str:
        """将对象转换为JSON字符串"""
        def default(o):
            if isinstance(o, datetime):
                return o.isoformat()
            return str(o)
            
        return json.dumps(obj, default=default, ensure_ascii=False)

# 创建全局JSON工具实例
json_utils = JsonUtils() 