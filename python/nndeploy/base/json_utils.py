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

# 测试代码
if __name__ == "__main__":
    import os
    import tempfile
    from datetime import datetime
    
    print("测试JSON工具模块")
    
    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test.json")
    
    # 测试数据
    test_data = {
        "name": "NNDeploy",
        "version": "1.0.0",
        "description": "深度学习模型部署框架",
        # "created_at": datetime.now(),
        "features": ["高性能", "易用性", "跨平台"],
        "config": {
            "backend": "pytorch",
            "device": "cuda",
            "options": {
                "precision": "fp16",
                "batch_size": 4
            }
        }
    }
    
    # 测试序列化和保存
    print("\n1. 测试JSON序列化和保存")
    json_str = JsonUtils.to_json_string(test_data)
    print(f"序列化结果: {json_str[:100]}...")
    
    JsonUtils.save(test_data, test_file)
    print(f"保存到文件: {test_file}")
    
    # 测试加载和解析
    print("\n2. 测试JSON加载和解析")
    loaded_data = JsonUtils.load(test_file)
    print(f"加载的数据: {loaded_data}")
    
    # 测试格式化
    print("\n3. 测试JSON格式化")
    formatted = JsonUtils.format_str(loaded_data, indent=2, sort_keys=True)
    print(f"格式化结果:\n{formatted[:200]}...")
    
    # 测试JSON Schema验证
    print("\n4. 测试JSON Schema验证")
    schema = {
        "type": "object",
        "required": ["name", "version"],
        "properties": {
            "name": {"type": "string"},
            "version": {"type": "string"},
            "features": {"type": "array"}
        }
    }
    
    is_valid = JsonUtils.is_valid(loaded_data, schema)
    print(f"数据验证结果: {'通过' if is_valid else '失败'}")
    
    # 测试无效数据
    invalid_data = {"name": 123, "version": "1.0"}
    is_valid = JsonUtils.is_valid(invalid_data, schema)
    print(f"无效数据验证结果: {'通过' if is_valid else '失败'}")
    
    # 测试合并
    print("\n5. 测试JSON合并")
    base = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
    update = {"b": 20, "c": {"e": 40, "f": 50}, "g": 60}
    
    merged = JsonUtils.merge(base, update)
    print(f"合并结果: {merged}")
    
    # 测试不覆盖合并
    merged_no_overwrite = JsonUtils.merge(base, update, overwrite=False)
    print(f"不覆盖合并结果: {merged_no_overwrite}")
    
    # 测试差异比较
    print("\n6. 测试JSON差异比较")
    diff = JsonUtils.diff(base, update)
    print(f"差异结果: {diff}")
    
    # 测试扁平化和还原
    print("\n7. 测试JSON扁平化和还原")
    nested = {
        "user": {
            "name": "张三",
            "profile": {
                "age": 30,
                "address": {
                    "city": "北京",
                    "district": "海淀"
                }
            }
        }
    }
    
    flattened = JsonUtils.flatten(nested)
    print(f"扁平化结果: {flattened}")
    
    unflattened = JsonUtils.unflatten(flattened)
    print(f"还原结果: {unflattened}")
    
    # 清理临时文件
    try:
        os.remove(test_file)
        os.rmdir(temp_dir)
        print("\n测试完成，临时文件已清理")
    except:
        print("\n测试完成，但临时文件清理失败")