import nndeploy._nndeploy_internal as _C

try:
    LlmInfer = _C.llm.LlmInfer
    StreamOut = _C.llm.StreamOut
except:
    pass