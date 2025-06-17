import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

try:
    TokenizerType = _C.tokenizer.TokenizerType
    TokenizerPraram = _C.tokenizer.TokenizerPraram
    TokenizerText = _C.tokenizer.TokenizerText
    TokenizerIds = _C.tokenizer.TokenizerIds
    TokenizerEncode = _C.tokenizer.TokenizerEncode
    TokenizerDecode = _C.tokenizer.TokenizerDecode
except:
    pass

try:
    TokenizerEncodeCpp = _C.tokenizer_cpp.TokenizerEncodeCpp
    TokenizerDecodeCpp = _C.tokenizer_cpp.TokenizerDecodeCpp
except:
    pass