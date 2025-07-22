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
    # 基础扩展
    'recommonmark',
    'sphinx_markdown_tables',
    
    # C++ API 扩展
    'breathe',
    
    # Python API 扩展
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
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
# 定义 Doxygen XML 的输出路径
# doxygen_xml_path = os.path.abspath("./build_doxygen/xml")

# breathe_projects = {
#     "nndeploy": doxygen_xml_path
# }
# breathe_default_project = "nndeploy"

doxygen_html_dir = os.path.abspath('./build_doxygen/html')
html_extra_path = [doxygen_html_dir]