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
from typing import Dict, List, Callable, Any
import json
import os
from pathlib import Path
import locale
from datetime import datetime
import flet as ft


class Language(Enum):
    """支持的语言枚举"""
    CHINESE = "zh"
    ENGLISH = "en"
    
    @classmethod
    def from_string(cls, value: str) -> 'Language':
        """从字符串创建语言枚举
        
        Args:
            value: 语言代码字符串
            
        Returns:
            对应的Language枚举值
            
        Raises:
            ValueError: 如果不是有效的语言代码
        """
        for lang in cls:
            if lang.value == value:
                return lang
        raise ValueError(f"不支持的语言代码: {value}")
    
    @classmethod
    def get_display_name(cls, language: 'Language', current_language: 'Language' = None) -> str:
        """获取语言的显示名称
        
        Args:
            language: 要获取显示名称的语言
            current_language: 当前使用的语言，决定返回的名称使用哪种语言
            
        Returns:
            语言的本地化显示名称
        """
        names = {
            cls.CHINESE: {"zh": "中文", "en": "Chinese"},
            cls.ENGLISH: {"zh": "英文", "en": "English"},
        }
        
        current = current_language or Language.ENGLISH
        return names.get(language, {}).get(current.value, language.value)


class LanguageManager:
    """语言管理类"""
    
    def __init__(self):
        # 尝试从系统获取默认语言
        system_locale = locale.getdefaultlocale()[0]
        default_lang = Language.ENGLISH  # 默认英文
        
        if system_locale:
            lang_code = system_locale.split('_')[0]
            try:
                default_lang = Language.from_string(lang_code)
            except ValueError:
                pass
                
        self._current_language = default_lang
        self._translations: Dict[str, Dict[str, str]] = {}
        self._observers: List[Callable[[Language], None]] = []
        self._load_translations()
        
    def _load_translations(self):
        """加载翻译文件"""
        base_path = Path(__file__).parent
        locales_path = base_path.parent / "assets/locales"
        # locales_path = Path("/home/always/github/public/nndeploy/python/nndeploy/ui/assets/locales")
        
        # 确保目录存在
        if not locales_path.exists():
            locales_path.mkdir(parents=True, exist_ok=True)
            print(f"创建语言资源目录: {locales_path}")
        
        # 初始化所有支持的语言
        for lang in Language:
            self._translations[lang.value] = {}
            
            file_path = locales_path / f"{lang.value}.json"
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        self._translations[lang.value] = json.load(f)
                except Exception as e:
                    print(f"加载语言文件 {file_path} 失败: {e}")
            else:
                print(f"语言文件不存在: {file_path}")
        # print(self._translations)

    def get_text(self, key: str, **kwargs) -> str:
        """获取指定key的当前语言文本
        
        Args:
            key: 文本键值,支持点号分隔的多级键值
            **kwargs: 文本中的变量值
            
        Returns:
            翻译后的文本,如果未找到则返回键值本身
        """
        translations = self._translations.get(self._current_language.value, {})
        # print(translations)
        
        # 支持多级键值访问
        # 直接查找键值
        text = translations.get(key, key)
        # print(text)
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
        
    def get_supported_languages(self) -> List[Language]:
        """获取所有支持的语言列表"""
        return list(Language)

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
            
    def save_translations(self, language: Language = None):
        """保存翻译到文件
        
        Args:
            language: 指定要保存的语言,如果为None则保存所有语言
        """
        base_path = Path(__file__).parent
        locales_path = base_path.parent / "assets/locales"
        
        # 确保目录存在
        if not locales_path.exists():
            locales_path.mkdir(parents=True, exist_ok=True)
            
        languages = [language] if language else Language
        
        for lang in languages:
            if isinstance(lang, Language):
                lang_value = lang.value
            else:
                lang_value = lang
                
            file_path = locales_path / f"{lang_value}.json"
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self._translations.get(lang_value, {}), f, 
                              ensure_ascii=False, indent=2)
                print(f"保存语言文件成功: {file_path}")
            except Exception as e:
                print(f"保存语言文件 {file_path} 失败: {e}")
                
    def add_translation(self, key: str, text: str, language: Language = None):
        """添加或更新翻译
        
        Args:
            key: 文本键值
            text: 翻译文本
            language: 目标语言,如果为None则使用当前语言
        """
        lang = language or self._current_language
        lang_value = lang.value if isinstance(lang, Language) else lang
        
        # 支持多级键值
        keys = key.split('.')
        translations = self._translations.setdefault(lang_value, {})
        
        # 遍历键值路径
        current = translations
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                # 最后一个键,设置值
                current[k] = text
            else:
                # 中间键,确保存在
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]


class FormatUtils:
    """格式化工具类"""
    
    @staticmethod
    def format_date(date: datetime, language: Language = None) -> str:
        """格式化日期
        
        Args:
            date: 日期对象
            language: 目标语言
            
        Returns:
            格式化后的日期字符串
        """
        if language == Language.CHINESE:
            return date.strftime("%Y年%m月%d日")
        else:
            return date.strftime("%b %d, %Y")
    
    @staticmethod
    def format_time(time: datetime, language: Language = None) -> str:
        """格式化时间
        
        Args:
            time: 时间对象
            language: 目标语言
            
        Returns:
            格式化后的时间字符串
        """
        return time.strftime("%H:%M:%S")
    
    @staticmethod
    def format_datetime(dt: datetime, language: Language = None) -> str:
        """格式化日期时间
        
        Args:
            dt: 日期时间对象
            language: 目标语言
            
        Returns:
            格式化后的日期时间字符串
        """
        date_str = FormatUtils.format_date(dt, language)
        time_str = FormatUtils.format_time(dt)
        
        if language == Language.CHINESE:
            return f"{date_str} {time_str}"
        else:
            return f"{date_str} at {time_str}"
    
    @staticmethod
    def format_number(number: float, group_separator: bool = True) -> str:
        """格式化数字
        
        Args:
            number: 数字
            group_separator: 是否使用千位分隔符
            
        Returns:
            格式化后的数字字符串
        """
        if group_separator:
            return f"{number:,}"
        else:
            return str(number)
    
    @staticmethod
    def format_decimal(number: float, decimals: int = 2) -> str:
        """格式化小数
        
        Args:
            number: 数字
            decimals: 小数位数
            
        Returns:
            格式化后的小数字符串
        """
        format_str = f"{{:.{decimals}f}}"
        return format_str.format(number)
    
    @staticmethod
    def format_percent(number: float, decimals: int = 0) -> str:
        """格式化百分比
        
        Args:
            number: 数字(0.01表示1%)
            decimals: 小数位数
            
        Returns:
            格式化后的百分比字符串
        """
        format_str = f"{{:.{decimals}f}}%"
        return format_str.format(number * 100)
    
    @staticmethod
    def format_file_size(size: int) -> str:
        """格式化文件大小
        
        Args:
            size: 文件大小(字节)
            
        Returns:
            格式化后的文件大小字符串
        """
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        size_float = float(size)
        unit_index = 0
        
        while size_float >= 1024 and unit_index < len(units) - 1:
            size_float /= 1024
            unit_index += 1
            
        return f"{FormatUtils.format_decimal(size_float, 2)} {units[unit_index]}"


# 创建全局语言管理器实例
language_manager = LanguageManager()

# 便捷函数
def get_text(key: str, **kwargs) -> str:
    """获取翻译文本的便捷函数
    
    Args:
        key: 文本键值
        **kwargs: 文本变量
        
    Returns:
        翻译后的文本
    """
    return language_manager.get_text(key, **kwargs)

def switch_language(language: Language):
    """切换语言的便捷函数
    
    Args:
        language: 目标语言
    """
    language_manager.switch_language(language)
    
def get_current_language() -> Language:
    """获取当前语言的便捷函数
    
    Returns:
        当前语言枚举值
    """
    return language_manager.get_current_language()
    
def get_supported_languages() -> List[Language]:
    """获取支持的语言列表的便捷函数
    
    Returns:
        语言枚举列表
    """
    return language_manager.get_supported_languages()

def format_date(date: datetime) -> str:
    """格式化日期的便捷函数"""
    return FormatUtils.format_date(date, get_current_language())

def format_time(time: datetime) -> str:
    """格式化时间的便捷函数"""
    return FormatUtils.format_time(time, get_current_language())

def format_datetime(dt: datetime) -> str:
    """格式化日期时间的便捷函数"""
    return FormatUtils.format_datetime(dt, get_current_language())

def format_number(number: float, group_separator: bool = True) -> str:
    """格式化数字的便捷函数"""
    return FormatUtils.format_number(number, group_separator)

def format_decimal(number: float, decimals: int = 2) -> str:
    """格式化小数的便捷函数"""
    return FormatUtils.format_decimal(number, decimals)

def format_percent(number: float, decimals: int = 0) -> str:
    """格式化百分比的便捷函数"""
    return FormatUtils.format_percent(number, decimals)

def format_file_size(size: int) -> str:
    """格式化文件大小的便捷函数"""
    return FormatUtils.format_file_size(size)


def main(page: ft.Page):
    """语言配置预览界面
    
    展示不同语言下的文本翻译效果，并提供语言切换功能
    """
    page.title = "语言配置预览"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    
    # 语言选择下拉框
    language_dropdown = ft.Dropdown(
        label="选择语言",
        width=200,
        options=[
            ft.dropdown.Option(text="中文", key=Language.CHINESE.value),
            ft.dropdown.Option(text="English", key=Language.ENGLISH.value),
        ],
        value=get_current_language().value,
    )
    
    # 翻译示例容器
    translation_container = ft.Column(spacing=10)
    
    # 更新翻译示例
    def update_translations():
        translation_container.controls.clear()
        
        # 添加各种翻译示例
        translation_container.controls.extend([
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("菜单项翻译示例", weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Text(f"文件菜单: {get_text('menu.file')}"),
                        ft.Text(f"新建工作流: {get_text('menu.file.new')}"),
                        ft.Text(f"打开工作流: {get_text('menu.file.open')}"),
                        ft.Text(f"保存: {get_text('menu.file.save')}"),
                    ]),
                    padding=15,
                )
            ),
            
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("对话框翻译示例", weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Text(f"确认: {get_text('dialog.confirm')}"),
                        ft.Text(f"取消: {get_text('dialog.cancel')}"),
                        ft.Text(f"关闭: {get_text('dialog.close')}"),
                    ]),
                    padding=15,
                )
            ),
            
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("格式化示例", weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Text(f"日期: {format_date(datetime.now())}"),
                        ft.Text(f"时间: {format_time(datetime.now())}"),
                        ft.Text(f"日期时间: {format_datetime(datetime.now())}"),
                        ft.Text(f"数字: {format_number(1234567.89)}"),
                        ft.Text(f"百分比: {format_percent(0.1234)}"),
                        ft.Text(f"文件大小: {format_file_size(1024*1024*3.5)}"),
                    ]),
                    padding=15,
                )
            ),
        ])
        
        page.update()
    
    # 语言切换处理函数
    def on_language_change(e):
        selected_language = Language(language_dropdown.value)
        switch_language(selected_language)
        update_translations()
    
    language_dropdown.on_change = on_language_change
    
    # 初始化界面
    update_translations()
    
    # 构建页面
    page.add(
        ft.Text("语言配置预览", size=24, weight=ft.FontWeight.BOLD),
        ft.Text("此页面展示了不同语言下的文本翻译效果，可通过下拉框切换语言"),
        language_dropdown,
        ft.Divider(),
        translation_container,
    )

# 如果直接运行此文件，则启动语言预览
if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=8080)
