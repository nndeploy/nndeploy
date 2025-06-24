import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

try:
    FairMotPreParam = _C.track.FairMotPreParam
    FairMotPostParam = _C.track.FairMotPostParam
    FairMotPreProcess = _C.track.FairMotPreProcess
    FairMotPostProcess = _C.track.FairMotPostProcess
    FairMotGraph = _C.track.FairMotGraph
except:
    pass