# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import shutil
import subprocess
import sphinx_rtd_theme

project = 'nndeploy'
copyright = 'nndeploy'
author = 'nndeploy'
release = '0.2.0'

# -- 编译和安装 nndeploy --------------------------------------------------
def build_and_install_nndeploy():
    """在conf.py中编译和安装nndeploy"""
    
    print("🚀 开始编译和安装 nndeploy...")
    
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    build_dir = os.path.join(project_root, 'build')
    python_dir = os.path.join(project_root, 'python')
    
    try:
        # # 1. 初始化子模块
        # print("📦 初始化子模块...")
        # subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], 
        #               cwd=project_root, check=True, capture_output=True)
        # print("✅ 子模块初始化完成")
        
        # # 2. 创建并进入build目录
        # print("🏗️  创建build目录...")
        # os.makedirs(build_dir, exist_ok=True)
        # # 拷贝cmake/config.cmake到build/config.cmake
        # config_src = os.path.join(project_root, 'cmake', 'config.cmake')
        # config_dst = os.path.join(build_dir, 'config.cmake')
        # shutil.copyfile(config_src, config_dst)
        # print("✅ 已拷贝 config.cmake 到 build 目录")
        
        # # 3. CMAKE配置
        # print("⚙️  执行CMAKE配置...")
        # cmake_cmd = ['cmake', '-DCMAKE_BUILD_TYPE=Release', '..']
        # subprocess.run(cmake_cmd, cwd=build_dir, check=True, capture_output=True)
        # print("✅ CMAKE配置完成")
        
        # # 4. 编译
        # print("🔨 开始编译...")
        # make_cmd = ['make', f'-j{os.cpu_count()}']
        # subprocess.run(make_cmd, cwd=build_dir, check=True, capture_output=True)
        # print("✅ 编译完成")
        
        # # 5. 安装
        # print("📦 执行make install...")
        # subprocess.run(['make', 'install'], cwd=build_dir, check=True, capture_output=True)
        # print("✅ make install完成")
        
        # 6. 设置库路径并安装Python包
        print("🐍 安装Python包...")
        
        # 设置环境变量
        library_path = os.path.join(python_dir, 'nndeploy')
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if current_ld_path:
            new_ld_path = f"{library_path}:{current_ld_path}"
        else:
            new_ld_path = library_path
        
        # 更新当前进程的环境变量
        os.environ['LD_LIBRARY_PATH'] = new_ld_path
        
        # 安装Python包
        pip_cmd = [sys.executable, '-m', 'pip', 'install', '-e', '.']
        subprocess.run(pip_cmd, cwd=python_dir, check=True, 
                      env=os.environ.copy(), capture_output=True)
        print("✅ Python包安装完成")
        
        # 7. 验证安装
        print("🔍 验证安装...")
        test_cmd = [sys.executable, '-c', 'import nndeploy; print(f"nndeploy version: {nndeploy.__version__}")']
        result = subprocess.run(test_cmd, cwd=python_dir, 
                               env=os.environ.copy(), capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ 验证成功: {result.stdout.strip()}")
            return True
        else:
            print(f"⚠️  验证失败: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 编译/安装过程出错: {e}")
        print(f"错误输出: {e.stderr if hasattr(e, 'stderr') and e.stderr else '无详细错误信息'}")
        return False
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return False

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

# 完善的Mock实现
class IntelligentMockModule:
    """智能Mock模块，能够返回Sphinx期望的数据类型"""
    
    def __init__(self, name='MockModule', return_value=None):
        self._name = name
        self._return_value = return_value
        
        # 预定义一些Sphinx和autodoc期望的属性
        self._mock_attributes = {
            '__all__': ['get_version', 'framework_init', 'framework_deinit', '__version__', 'get_type_enum_json'],
            '__name__': name,
            '__doc__': f'Mock module for {name}',
            '__file__': '/dev/null',
            '__path__': [],
            '__package__': name,
            '__version__': '0.2.0',
            '__loader__': None,
            '__spec__': None,
        }
    
    def __getattr__(self, name):
        # 返回预定义的属性
        if name in self._mock_attributes:
            return self._mock_attributes[name]
        
        # 对于函数调用，返回可调用的mock
        return IntelligentMockModule(f'{self._name}.{name}')
    
    def __call__(self, *args, **kwargs):
        # 使函数可调用
        return IntelligentMockModule(f'{self._name}()')
    
    def __str__(self):
        return f'<MockModule: {self._name}>'
    
    def __repr__(self):
        return self.__str__()
    
    def __getitem__(self, key):
        return IntelligentMockModule(f'{self._name}[{key}]')
    
    def __iter__(self):
        return iter([])
    
    def __len__(self):
        return 0

# 安全的模块导入函数
def safe_import_with_fallback():
    """安全导入nndeploy模块，失败时使用mock"""
    
    print("🔧 开始导入nndeploy模块...")
    
    # 检查是否存在C++扩展模块
    nndeploy_path = os.path.abspath('../../python')
    if nndeploy_path not in sys.path:
        sys.path.insert(0, nndeploy_path)
    
    try:
        import nndeploy._nndeploy_internal
        print("✅ nndeploy._nndeploy_internal模块导入成功")     
        # 尝试导入完整的nndeploy
        import nndeploy.dag
        print("✅ nndeploy模块导入成功")
        return True
        
    except ImportError as e:
        # 执行编译和安装
        build_success = build_and_install_nndeploy()
        if build_success:
            print("✅ 编译和安装nndeploy成功")
            return True
        else:
            print("❌ 编译和安装nndeploy失败")
            print(f"⚠️  导入失败: {e}")
            print("🔧 使用智能Mock模块...")

            # 创建mock模块层次结构
            mock_nndeploy = IntelligentMockModule('nndeploy')

            # 设置主模块
            sys.modules['nndeploy'] = mock_nndeploy
            sys.modules['nndeploy._nndeploy_internal'] = IntelligentMockModule('_nndeploy_internal')

            # 设置所有子模块
            submodules = [
                'base', 'device', 'ir', 'op', 'net', 'inference', 
                'dag', 'preprocess', 'tokenizer', 'codec', 
                'classification', 'detect', 'track', 'segment', 
                'matting', 'face', 'gan', '_C'
            ]

            for submodule in submodules:
                module_name = f'nndeploy.{submodule}'
                sys.modules[module_name] = IntelligentMockModule(module_name)

            # 为主要的nndeploy模块添加关键函数
            mock_nndeploy.get_version = lambda: "0.2.0"
            mock_nndeploy.framework_init = lambda: True
            mock_nndeploy.framework_deinit = lambda: True
            mock_nndeploy.get_type_enum_json = lambda: {}
            mock_nndeploy.__version__ = "0.2.0"

            print("✅ Mock模块设置完成")
            return False
        
           
# 执行安全导入
is_real_module = safe_import_with_fallback()

# autodoc配置 - 根据是否为真实模块调整
if is_real_module:
    print("📚 使用真实模块生成文档")
    autodoc_default_options = {
        'members': True,
        'undoc-members': True,
        'private-members': False,
        'special-members': '__init__',
        'inherited-members': True,
        'show-inheritance': True,
    }
else:
    print("📚 使用Mock模块生成文档")
    autodoc_default_options = {
        'members': False,          # 关闭自动成员检测
        'undoc-members': False,    # 关闭未文档化成员
        'private-members': False,
        'special-members': False,
        'inherited-members': False,
        'show-inheritance': False,
    }

# 其余配置保持不变...
autodoc_member_order = 'bysource'

# # autodoc 默认选项
# autodoc_default_options = {
#     'members': True,                # 包含所有公共成员
#     'undoc-members': True,          # 包含没有文档字符串的成员
#     'private-members': False,       # 不包含私有成员（如 _private）
#     'special-members': '__init__',  # 包含特殊成员（如 __init__）
#     'inherited-members': True,      # 包含继承的成员
#     'show-inheritance': True,       # 显示类继承关系
# }
# autodoc_member_order = 'bysource'   # 按源代码顺序排列成员

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
# print("🧬 运行 Doxygen...")
# try:
#     import subprocess
#     if os.path.exists('Doxyfile'):
#         subprocess.run(['doxygen'], check=True)
#         print("✅ Doxygen 完成")
#     else:
#         print("⚠️  Doxyfile 不存在，跳过")
# except:
#     print("❌ Doxygen 失败")
doxygen_html_dir = os.path.abspath('./build_doxygen/html')
# 删除doxygen_html_dir下index.html
if os.path.exists(os.path.join(doxygen_html_dir, 'index.html')):
    os.remove(os.path.join(doxygen_html_dir, 'index.html'))
html_extra_path = [doxygen_html_dir]