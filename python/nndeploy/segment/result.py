import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

try:
    SegmentResult = _C.segment.SegmentResult
except:
    pass