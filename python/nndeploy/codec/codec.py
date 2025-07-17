import nndeploy._nndeploy_internal as _C

Decode = _C.codec.Decode
Encode = _C.codec.Encode

create_decode_node = _C.codec.create_decode_node
create_encode_node = _C.codec.create_encode_node

OpenCvImageDecode = _C.codec.OpenCvImageDecode
OpenCvImagesDecode = _C.codec.OpenCvImagesDecode
OpenCvVideoDecode = _C.codec.OpenCvVideoDecode
OpenCvCameraDecode = _C.codec.OpenCvCameraDecode

OpenCvImageEncode = _C.codec.OpenCvImageEncode
OpenCvImagesEncode = _C.codec.OpenCvImagesEncode
OpenCvVideoEncode = _C.codec.OpenCvVideoEncode
OpenCvCameraEncode = _C.codec.OpenCvCameraEncode

create_opencv_decode = _C.codec.create_opencv_decode
create_opencv_encode = _C.codec.create_opencv_encode

BatchOpenCvDecode = _C.codec.BatchOpenCvDecode
BatchOpenCvEncode = _C.codec.BatchOpenCvEncode