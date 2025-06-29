import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

try:
    TokenizerType = _C.tokenizer.TokenizerType

    name_to_tokenizer_type = {
        "HF": _C.tokenizer.TokenizerType.kTokenizerTypeHF,
        "BPE": _C.tokenizer.TokenizerType.kTokenizerTypeBPE,
        "SentencePiece": _C.tokenizer.TokenizerType.kTokenizerTypeSentencePiece,
        "RWKVWorld": _C.tokenizer.TokenizerType.kTokenizerTypeRWKVWorld,
        "NotSupport": _C.tokenizer.TokenizerType.kTokenizerTypeNotSupport,
    }
    tokenizer_type_to_name = {v: k for k, v in name_to_tokenizer_type.items()}

    def get_tokenizer_type_enum_json():
        enum_list = []
        for tokenizer_type_name, tokenizer_type_code in name_to_tokenizer_type.items():
            tokenizer_type_str = _C.tokenizer.tokenizer_type_to_string(tokenizer_type_code)
            enum_list.append(tokenizer_type_str)
        tokenizer_type_enum = {}
        for single_enum in enum_list:
            tokenizer_type_enum[f"{single_enum}"] = enum_list
        return tokenizer_type_enum
    
    nndeploy.base.all_type_enum.append(get_tokenizer_type_enum_json)    
    
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