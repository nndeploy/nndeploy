

import nndeploy._nndeploy_internal as _C
from nndeploy.base import all_type_enum
from nndeploy.base import pretty_json_str
# from nndeploy.tokenizer import TokenizerEncode
import json
from nndeploy import get_type_enum_json
  
  
def test_enum_json():
    print("hello")
    print(json.dumps(get_type_enum_json(), indent=4))
    import nndeploy
    print(nndeploy)
    
    
if __name__ == "__main__":
    test_enum_json()