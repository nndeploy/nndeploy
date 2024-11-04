'''
函数形式Op
'''

import nndeploy._C as C


def rms_norm():
    return C.op.rms_norm()