import nndeploy._nndeploy_internal as _C

# 优化pass

FuseConvBias = _C.net.OptPassType.kOptPassTypeFuseConvBias
FuseConvBatchNorm = _C.net.OptPassType.kOptPassTypeFuseConvBatchNorm
FuseConvRelu = _C.net.OptPassType.kOptPassTypeFuseConvRelu
