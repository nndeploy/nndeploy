
import nndeploy

# 对应CPP的ModelDesc
class ModelDesc(nndeploy._C.ir.ModelDesc):
    
    def __init__(self) :
        super().__init__()