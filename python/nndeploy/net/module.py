from nndeploy.ir import ModelDesc
from nndeploy.base import DeviceType
from nndeploy.device import Tensor
import nndeploy._nndeploy_internal as _C

from .net import Net

from functools import wraps
import warnings

Expr = _C.op.Expr


def build_model(
    *, enable_static=False, enable_net_opt=True, enable_pass=set(), disable_pass=set()
):
    """
    类装饰器：用于 nndeploy.net.Module 的子类
    自动把 forward 方法装饰为：
        - 静态图：Expr → ModelDesc → Net
        - 动态图：Tensor → 直接计算
    """

    def _real_cls_decorator(cls):
        # 1. 生成新类名（可选）
        new_name = f"{cls.__name__}{'Static' if enable_static else 'Dynamic'}"

        # 2. 拷贝原类，防止污染
        NewClass = type(new_name, (cls,), {})

        # 3. 取出原 forward
        original_forward = cls.forward

        # 4. 用“函数装饰器”包装 original_forward
        @wraps(original_forward)
        def new_forward(self, *args, **kwargs):
            # ---------- 递归灌 weight_map ----------
            def _set_weight_map_recursive(module):
                for name in dir(module):
                    if name.startswith("_"):
                        continue
                    sub = getattr(module, name, None)
                    if not isinstance(sub, Module):
                        continue
                    # 静态图需要给子模块挂 model_desc
                    if enable_static:
                        sub.model_desc = self.model_desc
                    if hasattr(sub, "weight_map") and sub.weight_map is not None:
                        for k in sub.weight_map:
                            if k not in self.weight_map:
                                raise KeyError(k)
                            sub.weight_map[k] = self.weight_map[k]
                    _set_weight_map_recursive(sub)

            _set_weight_map_recursive(self)

            # ---------- 静态图分支 ----------
            if enable_static:
                if not all(isinstance(x, Expr) for x in args):
                    raise TypeError("静态图模式下输入必须是 Expr")
                result = original_forward(self, *args, **kwargs)
                # 标记输出
                if isinstance(result, (list, tuple)):
                    result = [_C.op.makeOutput(self.model_desc, r) for r in result]
                else:
                    result = _C.op.makeOutput(self.model_desc, result)
                # Net 初始化
                self.net.setModelDesc(self.model_desc)
                self.net.setDeviceType(self.device_type)
                self.net.enableOpt(enable_net_opt)
                self.net.setEnablePass(set(enable_pass))
                self.net.setDisablePass(set(disable_pass))
                if self.weight_map:
                    self.model_desc.set_weights(self.weight_map)
                else:
                    warnings.warn("weight_map is None")
                self.net.init()
                return result

            # ---------- 动态图分支 ----------
            else:
                if not all(isinstance(x, _C.device.Tensor) for x in args):
                    raise TypeError("动态图模式下输入必须是 Tensor")
                return original_forward(self, *args, **kwargs)

        # 5. 把新 forward 挂到新类
        NewClass.forward = new_forward
        return NewClass

    return _real_cls_decorator


class Module:

    def __init__(self):
        self.model_desc = ModelDesc()
        self.net = Net()
        self.device_type = DeviceType("cpu", 0)
        self.weight_map = None

    # 仅用于静态图的运行调用
    def run(self) -> list:
        self.net.preRun()
        self.net.run()
        self.net.postRun()
        return self.net.getAllOutput()

    def dump(self, file_path):
        self.net.dump(file_path)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def makeExpr(self, *args, **kwargs):
        raise NotImplementedError()

    def setInputs(self, inputs):
        self.net.setInputs(inputs)

    def __call__(self, *args, **kwargs):
        """
        严格路由：
        1. 所有输入都是 Expr  ->  静态图 makeExpr
        2. 所有输入都是 Tensor -> 动态图 forward
        其余情况 -> TypeError
        """
        flat_inputs = list(args) + list(kwargs.values())
        if not flat_inputs:  # 无参数防呆
            raise TypeError("At least one input is required")

        # 判断类别
        types = {type(x) for x in flat_inputs}
        if types == {Expr}:
            return self.makeExpr(*args, **kwargs)
        elif types == {Tensor}:
            return self.forward(*args, **kwargs)
        else:
            raise TypeError(
                f"{self.__class__.__name__} inputs must be all Tensor or all Expr"
            )
