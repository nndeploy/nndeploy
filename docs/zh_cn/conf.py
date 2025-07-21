# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import sphinx_rtd_theme

project = 'nndeploy'
copyright = 'nndeploy'
author = 'nndeploy'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'recommonmark',
    'sphinx_markdown_tables',
    'breathe',  # <--- 核心修复：在这里添加 breathe
] 

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'


locale_dirs = ['../locales/']
gettext_compact = False  # optional.
gettext_uuid = True  # optional.

 
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = ['../source/_templates']
html_static_path = ['../source/_static']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

# Python API 配置

# autodoc_mock_imports = [
#     'nndeploy._nndeploy_internal',
#     'nndeploy._C', # 也可能需要模拟整个 _C 包
# ]

extensions.extend([
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx.ext.viewcode',
        'sphinx.ext.napoleon',
        'sphinx.ext.intersphinx',
    ])

sys.path.insert(0, os.path.abspath('../../python'))

# autodoc 默认选项
autodoc_default_options = {
    'members': True,                # 包含所有公共成员
    'undoc-members': True,          # 包含没有文档字符串的成员
    'private-members': False,       # 不包含私有成员（如 _private）
    'special-members': '__init__',  # 包含特殊成员（如 __init__）
    'inherited-members': True,      # 包含继承的成员
    'show-inheritance': True,       # 显示类继承关系
}
autodoc_member_order = 'bysource'   # 按源代码顺序排列成员

# Napoleon 配置（用于解析 Google/NumPy 风格的 docstrings）
napoleon_google_docstring = True
napoleon_numpy_docstring = False # 通常只选择一种风格
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx 配置（链接外部文档）
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- C++ API 配置 -------------------------------------------------------------
xml_dir = os.path.abspath("./build/doxygen/xml")

breathe_projects = {
    "nndeploy_device": xml_dir
}
breathe_default_project = "nndeploy_device"
breathe_default_members = ('members', 'undoc-members')

# Breathe 显示配置
breathe_show_define_initializer = True
breathe_show_enumvalue_initializer = True

# 调试信息
breathe_show_define_initializer = True
breathe_show_enumvalue_initializer = True
# breathe_implementation_filename_extensions = ['.c', '.cc', '.cpp']
breathe_domain_by_extension = {
    "h": "cpp",
    "hpp": "cpp",
}

# 调试信息
print(f"🔧 Breathe 配置:")
print(f"   XML 目录: {xml_dir}")
print(f"   目录存在: {os.path.exists(xml_dir)}")
if os.path.exists(xml_dir):
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"   XML 文件数: {len(xml_files)}")
    if len(xml_files) > 0:
        print(f"   示例文件: {xml_files[:3]}")