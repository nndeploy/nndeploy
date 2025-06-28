import os
import sys

# 获取当前包的目录路径
package_dir = os.path.dirname(os.path.abspath(__file__))

# 根据不同平台设置库搜索路径
if sys.platform == 'win32':
    # Windows使用PATH环境变量
    if package_dir not in os.environ.get('PATH', '').split(';'):
        os.environ['PATH'] = f"{package_dir};{os.environ.get('PATH', '')}"
elif sys.platform == 'darwin':
    # macOS使用DYLD_LIBRARY_PATH环境变量
    if package_dir not in os.environ.get('DYLD_LIBRARY_PATH', '').split(':'):
        os.environ['DYLD_LIBRARY_PATH'] = f"{package_dir}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
else:
    # Linux使用LD_LIBRARY_PATH环境变量
    print(f"Linux: {package_dir}")
    if package_dir not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        os.environ['LD_LIBRARY_PATH'] = f"{package_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import nndeploy.base
import nndeploy.device
import nndeploy.ir
import nndeploy.op
import nndeploy.net
import nndeploy.inference
import nndeploy.dag
import nndeploy.preprocess
import nndeploy.tokenizer
import nndeploy.codec
import nndeploy.classification
import nndeploy.detect
import nndeploy.track
import nndeploy.segment
import nndeploy.matting

from .nndeploy import get_version, framework_init, framework_deinit
from .nndeploy import __version__
from .nndeploy import get_type_enum_json

__all__ = ['get_version', 'framework_init', 'framework_deinit', '__version__', 'get_type_enum_json']