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
release = '0.2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['recommonmark','sphinx_markdown_tables'] 

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
source_suffix = {'.rst': 'restructuredtext','.md': 'markdown'}

# python api
sys.path.insert(0, os.path.abspath('../../python/nndeploy'))  # 指向源码目录

# 自动发现所有模块
def find_all_modules():
    from pathlib import Path
    
    base_path = Path(os.path.abspath('../../python/nndeploy'))
    modules = []
    
    for path in base_path.glob('**/*.py'):
        if path.name == '__init__.py':
            module_path = path.relative_to(base_path).parent
        else:
            module_path = path.relative_to(base_path).with_suffix('')
        
        module_str = f"nndeploy.{str(module_path).replace('/', '.')}"
        
        if '__pycache__' not in module_str:
            modules.append(module_str)
    
    return sorted(set(modules))

# 在模板中暴露模块列表
def setup(app):
    app.add_config_value('all_modules', find_all_modules(), 'env')

# 扩展配置
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # 支持 Google/Numpy 风格
]

# 启用自动摘要
autosummary_generate = True
autosummary_imported_members = True

# 自动包含所有成员
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
    'special-members': '__init__',
}

# 自动为所有模块生成文档
autosummary_generate_overwrite = True