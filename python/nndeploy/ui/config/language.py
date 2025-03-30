"""
语言配置模块

负责:
- 定义UI文本的多语言映射
- 支持中英文动态切换
- 管理语言资源文件
- 提供语言切换接口

配置采用字典形式,支持运行时切换,切换时平滑过渡
"""

from enum import Enum
from typing import Dict, Optional, Callable
import json
import os
from pathlib import Path

class Language(Enum):
    """支持的语言枚举"""
    CHINESE = "zh"
    ENGLISH = "en"

class LanguageConfig:
    """语言配置类"""
    
    def __init__(self):
        self._current_language = Language.ENGLISH
        self._translations: Dict[str, Dict[str, str]] = {
            Language.CHINESE.value: {},
            Language.ENGLISH.value: {}
        }
        self._observers: List[Callable[[Language], None]] = []
        self._load_translations()

    def _load_translations(self):
        """加载翻译文件"""
        base_path = Path(__file__).parent
        # locales_path = base_path / "../assets/locales"
        locales_path_str = "/home/always/github/public/nndeploy/python/nndeploy/ui/assets/locales"  
        locales_path = Path(locales_path_str)
        
        for lang in Language:
            file_path = locales_path / f"{lang.value}.json"
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        self._translations[lang.value] = json.load(f)
                except Exception as e:
                    print(f"加载语言文件 {file_path} 失败: {e}")

    def get_text(self, key: str, **kwargs) -> str:
        """获取指定key的当前语言文本
        
        Args:
            key: 文本键值,支持点号分隔的多级键值
            **kwargs: 文本中的变量值
            
        Returns:
            翻译后的文本,如果未找到则返回键值本身
        """
        translations = self._translations[self._current_language.value]
        
        # 支持多级键值访问
        if key in translations:
            text = translations[key]
        else:
            # print(f"key not found: {key}")
            text = key
                
        # 替换文本中的变量
        if isinstance(text, str) and kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass
                
        return text

    def switch_language(self, language: Language):
        """切换语言
        
        Args:
            language: 目标语言
            
        Raises:
            ValueError: 如果指定了不支持的语言
        """
        if not isinstance(language, Language):
            raise ValueError(f"不支持的语言: {language}")
            
        if language != self._current_language:
            self._current_language = language
            # 通知所有观察者语言已更改
            for observer in self._observers:
                observer(language)

    def get_current_language(self) -> Language:
        """获取当前语言设置"""
        return self._current_language

    def add_observer(self, observer: Callable[[Language], None]):
        """添加语言变化观察者
        
        Args:
            observer: 观察者回调函数,接收新的语言枚举值作为参数
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: Callable[[Language], None]):
        """移除语言变化观察者
        
        Args:
            observer: 要移除的观察者回调函数
        """
        if observer in self._observers:
            self._observers.remove(observer)

# 创建全局语言配置实例
language_config = LanguageConfig()

# 便捷函数用于获取翻译文本
def get_text(key: str, **kwargs) -> str:
    """获取翻译文本的便捷函数
    
    Args:
        key: 文本键值
        **kwargs: 文本变量
        
    Returns:
        翻译后的文本
    """
    return language_config.get_text(key, **kwargs) 