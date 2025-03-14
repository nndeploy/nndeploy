"""
快捷键配置模块测试

测试内容:
- 快捷键配置加载
- 快捷键触发回调
- 快捷键表格UI展示
- 快捷键自定义设置
"""

import sys
import os
from pathlib import Path
import flet as ft

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from nndeploy.ui.config.shortcuts import (
    ShortcutCategory, 
    ModifierKey, 
    shortcut_config, 
    register_page, 
    set_callback,
    create_shortcuts_table
)

def main(page: ft.Page):
    """使用Flet应用程序测试快捷键配置模块"""
    page.title = "快捷键配置模块测试"
    page.padding = 20
    page.scroll = "auto"
    
    # 注册页面以启用快捷键功能
    register_page(page)
    
    # 创建标题
    title = ft.Text("快捷键配置模块测试", size=30, weight=ft.FontWeight.BOLD)
    
    # 创建测试结果显示区域
    result_text = ft.Text("", selectable=True)
    
    # 创建测试按钮
    def update_result(text):
        result_text.value = text
        page.update()
    
    # 测试快捷键配置加载
    def test_shortcuts_loading(e):
        output = []
        
        # 获取所有分类的快捷键
        for category in ShortcutCategory:
            shortcuts = shortcut_config.get_shortcuts_by_category(category)
            output.append(f"分类 {category.value} 的快捷键数量: {len(shortcuts)}")
            
            # 显示前3个快捷键
            for i, shortcut in enumerate(shortcuts[:3]):
                output.append(f"  {i+1}. {shortcut.description}: {str(shortcut)}")
        
        update_result("\n".join(output))
    
    # 测试快捷键回调
    def test_shortcut_callback(e):
        output = ["快捷键回调测试:"]
        
        # 定义回调函数
        def on_shortcut_triggered(shortcut_id):
            output.append(f"快捷键 '{shortcut_id}' 被触发!")
            update_result("\n".join(output))
        
        # 为几个常用快捷键设置回调
        shortcut_ids = ["new_workflow", "save_workflow", "undo", "redo"]
        for shortcut_id in shortcut_ids:
            success = set_callback(
                shortcut_id, 
                lambda id=shortcut_id: on_shortcut_triggered(id)
            )
            if success:
                shortcut = shortcut_config.get_shortcut(shortcut_id)
                output.append(f"已为 '{shortcut_id}' ({str(shortcut)}) 设置回调")
            else:
                output.append(f"为 '{shortcut_id}' 设置回调失败")
        
        output.append("\n请尝试按下以下快捷键:")
        for shortcut_id in shortcut_ids:
            shortcut = shortcut_config.get_shortcut(shortcut_id)
            if shortcut:
                output.append(f"- {str(shortcut)} ({shortcut.description})")
        
        update_result("\n".join(output))
    
    # 测试快捷键表格UI
    def test_shortcuts_table(e):
        # 清除之前的表格
        for control in page.controls[:]:
            if isinstance(control, ft.DataTable):
                page.controls.remove(control)
        
        # 创建快捷键表格
        table = create_shortcuts_table()
        
        # 添加表格到页面
        page.add(table)
        page.update()
        
        update_result("已显示快捷键表格。您可以查看所有可用的快捷键。")
    
    # 测试快捷键冲突检测
    def test_shortcut_conflict(e):
        output = ["快捷键冲突检测测试:"]
        
        # 尝试设置一个已存在的快捷键组合
        existing_shortcut = shortcut_config.get_shortcut("save_workflow")
        if existing_shortcut:
            # 尝试将"new_workflow"设置为与"save_workflow"相同的快捷键
            success = shortcut_config.set_shortcut(
                "new_workflow",
                existing_shortcut.key,
                existing_shortcut.modifiers,
                save=False  # 不保存到配置文件
            )
            
            if success:
                output.append("错误: 允许设置冲突的快捷键!")
            else:
                output.append("成功: 检测到快捷键冲突并阻止设置")
                
            # 恢复原始快捷键
            original = shortcut_config.get_shortcut("new_workflow")
            if original:
                shortcut_config.set_shortcut(
                    "new_workflow",
                    original.key,
                    original.modifiers,
                    save=False
                )
        else:
            output.append("无法找到'save_workflow'快捷键进行测试")
        
        update_result("\n".join(output))
    
    # 创建按钮
    load_btn = ft.ElevatedButton("测试快捷键配置加载", on_click=test_shortcuts_loading)
    callback_btn = ft.ElevatedButton("测试快捷键回调", on_click=test_shortcut_callback)
    table_btn = ft.ElevatedButton("显示快捷键表格", on_click=test_shortcuts_table)
    conflict_btn = ft.ElevatedButton("测试快捷键冲突检测", on_click=test_shortcut_conflict)
    
    # 添加控件到页面
    page.add(
        title,
        ft.Row([load_btn, callback_btn, table_btn, conflict_btn]),
        ft.Divider(),
        ft.Text("测试结果:", weight=ft.FontWeight.BOLD),
        result_text
    )

if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=9090)
