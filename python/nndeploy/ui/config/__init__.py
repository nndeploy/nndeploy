"""Configuration package"""
from .language import language_config, get_text, Language
from .theme import theme_config, get_color, get_style, ThemeType
from .settings import settings, get_setting, set_setting
from .shortcuts import shortcut_config, get_shortcut, get_shortcuts_by_category

__all__ = [
    'language_config',
    'get_text',
    'Language',
    'theme_config',
    'get_color',
    'get_style', 
    'ThemeType',
    'settings',
    'get_setting',
    'set_setting',
    'shortcut_config',
    'get_shortcut',
    'get_shortcuts_by_category'
] 