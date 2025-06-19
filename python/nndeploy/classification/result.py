import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

try:
    ClassificationLableResult = _C.classification.ClassificationLableResult
    ClassificationResult = _C.classification.ClassificationResult
except:
    pass