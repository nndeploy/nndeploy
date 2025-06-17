import nndeploy._nndeploy_internal as _C

DecodeNode = _C.codec.DecodeNode
EncodeNode = _C.codec.EncodeNode

create_decode_node = _C.codec.create_decode_node
create_encode_node = _C.codec.create_encode_node

OpenCvImageDecodeNode = _C.codec.OpenCvImageDecodeNode
OpenCvImageEncodeNode = _C.codec.OpenCvImageEncodeNode

BatchOpenCvDecode = _C.codec.BatchOpenCvDecode
BatchOpenCvEncode = _C.codec.BatchOpenCvEncode