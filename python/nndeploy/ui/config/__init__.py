
from .settings import get_setting, set_setting

from .theme import ThemeType, get_color, get_style, switch_theme, get_current_theme, set_custom_color, set_custom_style, get_flet_theme

from .language import Language, get_text, switch_language, get_current_language, get_supported_languages, format_date, format_time, format_datetime, format_number, format_percent, format_file_size

from .shortcuts import ModifierKey, ShortcutCategory, Shortcut, get_shortcut, get_shortcuts_by_category, register_page, set_callback, create_shortcuts_table

