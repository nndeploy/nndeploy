

import nndeploy._nndeploy_internal as _C
from nndeploy.base import all_type_enum
from nndeploy.base import pretty_json_str
# from nndeploy.tokenizer import TokenizerEncode
import json
from nndeploy import get_type_enum_json
  
  
def test_enum_json():
    json_str = json.dumps(get_type_enum_json(), indent=4)
    with open("all_type_enum.json", "w") as f:
        f.write(json_str)
    
    
if __name__ == "__main__":
    test_enum_json()