import nndeploy._nndeploy_internal as _C

Decode = _C.codec.Decode
EncodeNode = _C.codec.EncodeNode

create_decode_node = _C.codec.create_decode_node
create_encode_node = _C.codec.create_encode_node

OpenCvImageDecode = _C.codec.OpenCvImageDecode
OpenCvImageEncodeNode = _C.codec.OpenCvImageEncodeNode

BatchOpenCvDecode = _C.codec.BatchOpenCvDecode
BatchOpenCvEncode = _C.codec.BatchOpenCvEncode