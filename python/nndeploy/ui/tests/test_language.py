"""
语言配置模块测试

测试内容:
- 语言切换功能
- 文本翻译获取
- 观察者模式
"""

import sys
import os
from pathlib import Path
import flet as ft

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from nndeploy.ui.config.language import Language, LanguageConfig, language_config, get_text

def main(page: ft.Page):
    """使用Flet应用程序测试语言配置模块"""
    page.title = "语言配置模块测试"
    page.padding = 20
    page.scroll = "auto"
    
    # 创建标题
    title = ft.Text("语言配置模块测试", size=30, weight=ft.FontWeight.BOLD)
    
    # 创建测试结果显示区域
    result_text = ft.Text("", selectable=True)
    
    # 创建测试按钮
    def update_result(text):
        result_text.value = text
        page.update()
    
    # 测试默认语言
    def test_default_language(e):
        current_language = language_config.get_current_language()
        update_result(f"当前默认语言: {current_language}")
    
    # 测试语言切换
    def test_switch_language(e):
        output = []
        # 获取当前语言
        current_language = language_config.get_current_language()
        output.append(f"当前语言: {current_language}")
        
        # 切换语言
        new_language = Language.ENGLISH if current_language == Language.CHINESE else Language.CHINESE
        language_config.switch_language(new_language)
        output.append(f"切换后语言: {language_config.get_current_language()}")
        
        update_result("\n".join(output))
    
    # 测试获取翻译文本
    def test_get_text(e):
        output = []
        # 测试中文
        language_config.switch_language(Language.CHINESE)
        output.append(f"当前语言: {language_config.get_current_language()}")
        
        # 测试一些常用翻译键
        keys = ["dialog.confirm", "dialog.cancel", "menu.file.save"]
        for key in keys:
            text = get_text(key)
            output.append(f"键: {key}, 中文翻译: {text}")
        
        # 切换到英文
        language_config.switch_language(Language.ENGLISH)
        output.append(f"当前语言: {language_config.get_current_language()}")
        
        # 测试相同的键
        for key in keys:
            text = get_text(key)
            output.append(f"键: {key}, 英文翻译: {text}")
        
        # 测试变量替换
        text_with_var = get_text("common.welcome", name="测试用户")
        output.append(f"带变量的翻译: {text_with_var}")
        
        # 测试不存在的键
        non_existent_key = "this.key.does.not.exist"
        output.append(f"不存在的键: {non_existent_key}, 翻译结果: {get_text(non_existent_key)}")
        
        update_result("\n".join(output))
    
    # 测试观察者模式
    language_changed = False
    new_language = None
    
    def language_observer(language):
        nonlocal language_changed, new_language
        language_changed = True
        new_language = language
        update_result(f"语言已更改为: {language}")
    
    def test_observer_pattern(e):
        nonlocal language_changed, new_language
        output = []
        
        # 添加观察者
        language_config.add_observer(language_observer)
        output.append("已添加语言观察者")
        
        # 切换语言
        current_language = language_config.get_current_language()
        new_lang = Language.ENGLISH if current_language == Language.CHINESE else Language.CHINESE
        language_config.switch_language(new_lang)
        
        # 验证观察者被调用
        if language_changed:
            output.append(f"观察者被调用，新语言: {new_language}")
        else:
            output.append("观察者未被调用")
        
        # 重置状态
        language_changed = False
        new_language = None
        
        # 移除观察者
        language_config.remove_observer(language_observer)
        output.append("已移除语言观察者")
        
        # 再次切换语言
        language_config.switch_language(current_language)
        
        # 验证观察者未被调用
        if not language_changed:
            output.append("移除后观察者未被调用")
        else:
            output.append("移除后观察者仍被调用，测试失败")
        
        update_result("\n".join(output))
    
    # 创建按钮
    default_language_btn = ft.ElevatedButton("测试默认语言", on_click=test_default_language)
    switch_language_btn = ft.ElevatedButton("测试语言切换", on_click=test_switch_language)
    get_text_btn = ft.ElevatedButton("测试获取翻译文本", on_click=test_get_text)
    observer_btn = ft.ElevatedButton("测试观察者模式", on_click=test_observer_pattern)
    
    # 添加控件到页面
    page.add(
        title,
        ft.Row([default_language_btn, switch_language_btn, get_text_btn, observer_btn]),
        ft.Divider(),
        ft.Text("测试结果:", weight=ft.FontWeight.BOLD),
        result_text
    )

if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=9090)
