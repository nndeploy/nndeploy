
import nndeploy._nndeploy_internal as _C

def get_version():
    return _C.get_version()

def framework_init():
    return _C.nndeployFrameworkInit()

def framework_deinit():
    return _C.nndeployFrameworkDeinit()
  
  
__version__ = get_version()