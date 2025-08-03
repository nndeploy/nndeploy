import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

try:
    ClassificationPostParam = _C.classification.ClassificationPostParam
    ClassificationPostProcess = _C.classification.ClassificationPostProcess
    ClassificationGraph = _C.classification.ClassificationGraph
    ResnetGraph = _C.classification.ResnetGraph
except:
    pass