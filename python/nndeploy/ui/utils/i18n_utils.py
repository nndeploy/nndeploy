"""
国际化工具模块

负责:
- 处理多语言文本转换
- 提供日期时间本地化
- 处理数字格式本地化
- 管理语言资源文件

国际化文本无缝集成到UI中
"""

import locale
from datetime import datetime
from typing import Any, Dict, Optional
import json
from pathlib import Path
from babel import dates, numbers
from babel.dates import format_date, format_time, format_datetime
from babel.numbers import format_number, format_decimal, format_percent

class I18nUtils:
    """国际化工具类"""
    
    def __init__(self):
        self._locale = locale.getdefaultlocale()[0]
        self._translations: Dict[str, Dict[str, str]] = {}
        self._load_translations()
        
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
                    print(f"加载语言文件 {file_path} 失败: {e}")
                    
    def set_locale(self, locale_str: str):
        """设置当前语言环境"""
        self._locale = locale_str
        
    def get_locale(self) -> str:
        """获取当前语言环境"""
        return self._locale
        
    def translate(self, key: str, locale: str = None, **kwargs) -> str:
        """翻译文本
        
        Args:
            key: 文本键值
            locale: 目标语言(可选)
            **kwargs: 文本变量
            
        Returns:
            翻译后的文本
        """
        locale = locale or self._locale
        lang = locale.split("_")[0]
        
        # 获取翻译文本
        text = self._translations.get(lang, {}).get(key, key)
        
        # 替换变量
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass
                
        return text
        
    def format_date(
        self,
        date: datetime,
        format: str = "medium",
        locale: str = None
    ) -> str:
        """格式化日期
        
        Args:
            date: 日期对象
            format: 格式(full/long/medium/short)
            locale: 目标语言(可选)
        """
        locale = locale or self._locale
        return format_date(date, format=format, locale=locale)
        
    def format_time(
        self,
        time: datetime,
        format: str = "medium",
        locale: str = None
    ) -> str:
        """格式化时间"""
        locale = locale or self._locale
        return format_time(time, format=format, locale=locale)
        
    def format_datetime(
        self,
        dt: datetime,
        format: str = "medium",
        locale: str = None
    ) -> str:
        """格式化日期时间"""
        locale = locale or self._locale
        return format_datetime(dt, format=format, locale=locale)
        
    def format_number(
        self,
        number: float,
        locale: str = None,
        group_separator: bool = True
    ) -> str:
        """格式化数字"""
        locale = locale or self._locale
        return format_number(
            number,
            locale=locale,
            group_separator=group_separator
        )
        
    def format_decimal(
        self,
        number: float,
        decimals: int = 2,
        locale: str = None
    ) -> str:
        """格式化小数"""
        locale = locale or self._locale
        return format_decimal(
            number,
            format=f"#,##0.{'0' * decimals}",
            locale=locale
        )
        
    def format_percent(
        self,
        number: float,
        decimals: int = 0,
        locale: str = None
    ) -> str:
        """格式化百分比"""
        locale = locale or self._locale
        return format_percent(
            number,
            format=f"#,##0.{'0' * decimals}%",
            locale=locale
        )
        
    def format_file_size(
        self,
        size: int,
        locale: str = None
    ) -> str:
        """格式化文件大小"""
        locale = locale or self._locale
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{format_decimal(size, format='#,##0.##', locale=locale)} {unit}"
            size /= 1024
        return f"{format_decimal(size, format='#,##0.##', locale=locale)} PB"

# 创建全局国际化工具实例
i18n = I18nUtils()

# 便捷函数
def translate(key: str, locale: str = None, **kwargs) -> str:
    return i18n.translate(key, locale, **kwargs)

def format_date(date: datetime, format: str = "medium", locale: str = None) -> str:
    return i18n.format_date(date, format, locale)

def format_time(time: datetime, format: str = "medium", locale: str = None) -> str:
    return i18n.format_time(time, format, locale)

def format_datetime(dt: datetime, format: str = "medium", locale: str = None) -> str:
    return i18n.format_datetime(dt, format, locale)

def format_number(number: float, locale: str = None, group_separator: bool = True) -> str:
    return i18n.format_number(number, locale, group_separator)

def format_decimal(number: float, decimals: int = 2, locale: str = None) -> str:
    return i18n.format_decimal(number, decimals, locale)

def format_percent(number: float, decimals: int = 0, locale: str = None) -> str:
    return i18n.format_percent(number, decimals, locale)

def format_file_size(size: int, locale: str = None) -> str:
    return i18n.format_file_size(size, locale) 