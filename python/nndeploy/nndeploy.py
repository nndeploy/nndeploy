
import nndeploy._nndeploy_internal as _C
from nndeploy.base import all_type_enum
import json


def get_version():
    return _C.get_version()


def framework_init():
    return _C.nndeployFrameworkInit()


def framework_deinit():
    return _C.nndeployFrameworkDeinit()
  
  
__version__ = get_version()


def get_type_enum_json():
    all_type_enum_json = {}
    for type_enum in all_type_enum:
        all_type_enum_json.update(type_enum())
    return all_type_enum_json