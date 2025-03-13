"""
语言服务模块

负责:
- 处理UI文本的多语言切换
- 管理语言资源文件
- 记录用户语言偏好
- 提供实时翻译功能

语言切换平滑过渡,保持操作上下文
"""

from typing import Dict, List, Optional, Callable
import json
from pathlib import Path
import locale

from ..utils.logger import logger
from ..utils.i18n_utils import I18nUtils

class Language:
    """语言定义"""
    CHINESE = "zh"
    ENGLISH = "en"
    
    @staticmethod
    def get_display_name(lang_code: str) -> str:
        """获取语言显示名称"""
        names = {
            Language.CHINESE: "简体中文",
            Language.ENGLISH: "English"
        }
        return names.get(lang_code, lang_code)

class LanguageService:
    """语言服务类"""
    
    def __init__(self):
        self._current_language = self._detect_language()
        self._translations: Dict[str, Dict[str, str]] = {}
        self._observers = []
        self._load_translations()
        
    def _detect_language(self) -> str:
        """检测系统语言"""
        try:
            system_lang = locale.getdefaultlocale()[0]
            if system_lang:
                if system_lang.startswith("zh"):
                    return Language.CHINESE
                elif system_lang.startswith("en"):
                    return Language.ENGLISH
        except Exception as e:
            logger.error(f"检测系统语言失败: {e}")
            
        return Language.ENGLISH
        
    def _load_translations(self):
        """加载翻译文件"""
        locales_dir = Path(__file__).parent.parent / "assets/locales"
        
        if locales_dir.exists():
            for file_path in locales_dir.glob("*.json"):
                lang = file_path.stem
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        self._translations[lang] = json.load(f)
                except Exception as e:
                    logger.error(f"加载语言文件 {file_path} 失败: {e}")
                    
    def get_current_language(self) -> str:
        """获取当前语言"""
        return self._current_language
        
    def get_available_languages(self) -> List[str]:
        """获取可用语言列表"""
        return list(self._translations.keys())
        
    def switch_language(self, language: str):
        """切换语言
        
        Args:
            language: 目标语言代码
        """
        if language not in self._translations:
            raise ValueError(f"不支持的语言: {language}")
            
        if language != self._current_language:
            self._current_language = language
            # 通知观察者
            for observer in self._observers:
                observer(language)
                
    def get_text(self, key: str, **kwargs) -> str:
        """获取翻译文本
        
        Args:
            key: 文本键值
            **kwargs: 文本变量
            
        Returns:
            翻译后的文本
        """
        # 获取翻译文本
        text = self._translations.get(self._current_language, {}).get(key, key)
        
        # 替换变量
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass
                
        return text
        
    def add_observer(self, observer: Callable[[str], None]):
        """添加语言变化观察者"""
        if observer not in self._observers:
            self._observers.append(observer)
            
    def remove_observer(self, observer: Callable[[str], None]):
        """移除语言变化观察者"""
        if observer in self._observers:
            self._observers.remove(observer)

# 创建全局语言服务实例
language_service = LanguageService()

# 便捷函数
def get_text(key: str, **kwargs) -> str:
    return language_service.get_text(key, **kwargs) 