from nndeploy import _nndeploy_internal as _C

if hasattr(_C, "matting"):
    from .result import MattingResult
    from .pp_matting import PPMattingGraph, PPMattingPostParam, PPMattingPostProcess
    from .vis_matting import VisMatting
