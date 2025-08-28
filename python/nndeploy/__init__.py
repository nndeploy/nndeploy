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
        
def _setup_symlinks():
    """Rebuild symlinks during import"""
    import os
    import json
    import platform
    # print(f"platform.system(): {platform.system()}")
    
    # Only handle symlinks on Linux and macOS
    if platform.system() == "Windows":
        return
    
    package_dir = os.path.dirname(__file__)
    symlink_info_file = os.path.join(package_dir, "_symlink_info.json")
    
    if not os.path.exists(symlink_info_file):
        return
    
    try:
        with open(symlink_info_file, 'r') as f:
            symlink_info = json.load(f)
        
        for link_name, target_name in symlink_info.items():
            link_path = os.path.join(package_dir, link_name)
            target_path = os.path.join(package_dir, target_name)
            
            # Check if target file exists
            if not os.path.exists(target_path):
                print(f"Warning: Target file {target_name} not found for symlink {link_name}")
                continue
            
            # Skip if symlink already exists and is correct
            if os.path.islink(link_path) and os.readlink(link_path) == target_name:
                continue
            
            # Remove existing file (may be incorrect symlink or regular file)
            if os.path.exists(link_path) or os.path.islink(link_path):
                os.unlink(link_path)
            
            # Create symlink
            os.symlink(target_name, link_path)
            print(f"Created symlink: {link_name} -> {target_name}")
            
    except Exception as e:
        print(f"Warning: Failed to setup symlinks: {e}")

# Execute automatically during module import
_setup_symlinks()

import nndeploy.base
import nndeploy.device
import nndeploy.ir
import nndeploy.op
import nndeploy.net
import nndeploy.inference
import nndeploy.dag
import nndeploy.basic
import nndeploy.preprocess
import nndeploy.tokenizer
import nndeploy.codec
import nndeploy.classification
import nndeploy.detect
import nndeploy.track
import nndeploy.segment
import nndeploy.matting
import nndeploy.face
import nndeploy.gan
# import nndeploy.diffusers

from .nndeploy import get_version, framework_init, framework_deinit
from .nndeploy import __version__
from .nndeploy import get_type_enum_json

__all__ = ['get_version', 'framework_init', 'framework_deinit', '__version__', 'get_type_enum_json']