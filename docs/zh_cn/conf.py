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

# # Python API 配置
# extensions.extend([
#         'sphinx.ext.autodoc',
#         'sphinx.ext.autosummary',
#         'sphinx.ext.viewcode',
#         'sphinx.ext.napoleon',
#         'sphinx.ext.intersphinx',
#     ])

# sys.path.insert(0, os.path.abspath('../../python/nndeploy'))

# # autodoc 配置
# autodoc_default_options = {
#     'members': True,                # 包含所有成员
#     'undoc-members': True,          # 包含无文档字符串的成员
#     'private-members': False,       # 排除私有成员
#     'special-members': '__init__',  # 包含特殊方法
#     'inherited-members': True,      # 包含继承的成员
#     'show-inheritance': True,       # 显示继承关系
# }

# # autosummary 配置
# autosummary_generate = True              # 自动生成摘要文件
# autosummary_generate_overwrite = True    # 覆盖已存在的文件
# autosummary_imported_members = True      # 包含导入的成员

# # Napoleon 配置（docstring 风格）
# napoleon_google_docstring = True         # 支持 Google 风格
# napoleon_numpy_docstring = True          # 支持 NumPy 风格
# napoleon_include_init_with_doc = False   # __init__ 文档处理
# napoleon_include_private_with_doc = False
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True

# # Intersphinx 配置（链接外部文档）
# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3/', None),
#     'numpy': ('https://numpy.org/doc/stable/', None),
#     'torch': ('https://pytorch.org/docs/stable/', None),
# }

# # -- C++ API 配置 -------------------------------------------------------------
# extensions.extend([
#     'breathe',
#     'exhale',
#     ])

# # Breathe 配置
# breathe_projects = {
#     "nndeploy": "build/doxygen/xml"
# }
# breathe_default_project = "nndeploy"
# breathe_default_members = ('members', 'undoc-members')

# # Exhale 配置
# exhale_args = {
#     # 输出目录
#     "containmentFolder": "./cpp_api",
#     "rootFileName": "library_root.rst",
#     "rootFileTitle": "C++ API 参考",
    
#     # Doxygen 配置
#     "doxygenStripFromPath": "../..",
#     "createTreeView": True,
#     "exhaleExecutesDoxygen": True,
    
#     # 自定义 Doxygen 配置
#     "exhaleDoxygenStdin": """
#         PROJECT_NAME           = "nndeploy"
#         PROJECT_NUMBER         = "0.2.0"
#         PROJECT_BRIEF          = "高性能神经网络部署框架"
        
#         # 输入路径
#         INPUT                  = ../../framework
#         RECURSIVE              = YES
#         FILE_PATTERNS          = *.h *.hpp *.cpp *.cc *.c
        
#         # 排除路径
#         EXCLUDE_PATTERNS       = */third_party/* */build/* */.git/* */test/*
        
#         # 提取配置
#         EXTRACT_ALL            = YES
#         EXTRACT_PRIVATE        = NO
#         EXTRACT_STATIC         = YES
#         EXTRACT_LOCAL_CLASSES  = YES
#         EXTRACT_ANON_NSPACES   = NO
        
#         # 输出配置
#         GENERATE_HTML          = NO
#         GENERATE_XML           = YES
#         XML_OUTPUT             = ../build/doxygen/xml
#         XML_PROGRAMLISTING     = YES
        
#         # 预处理配置
#         ENABLE_PREPROCESSING   = YES
#         MACRO_EXPANSION        = YES
#         EXPAND_ONLY_PREDEF     = YES
        
#         # 图形配置
#         HAVE_DOT               = YES
#         DOT_IMAGE_FORMAT       = svg
#         DOT_TRANSPARENT        = YES
#         CALL_GRAPH             = YES
#         CALLER_GRAPH           = YES
        
#         # 其他配置
#         QUIET                  = YES
#         WARNINGS               = YES
#         WARN_IF_UNDOCUMENTED   = NO
#         """,
# }