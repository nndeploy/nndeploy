from nndeploy.ir import ModelDesc
from nndeploy.base import DeviceType
from nndeploy.device import Tensor
import nndeploy._nndeploy_internal as _C

from .net import Net

from functools import wraps
import warnings

Expr = _C.op.Expr


def build_model(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        # 递归地给所有子模块（含多级嵌套）设置 weight_map
        def _set_weight_map_recursive(module):
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue  # 跳过私有/保护成员
                submodule = getattr(module, attr_name, None)
                if not isinstance(submodule, Module):
                    continue  # 不是 Module，跳过
                submodule.model_desc = self.model_desc
                # 如果当前子模块自己有 weight_map，就赋值
                if hasattr(submodule, "weight_map"):
                    sub_wm = getattr(submodule, "weight_map")
                    if sub_wm is not None:
                        for k in sub_wm:
                            if k not in self.weight_map:
                                raise KeyError(
                                    f"weight '{k}' is not initialized in Model.weight_map"
                                )
                            sub_wm[k] = self.weight_map[k]

                # 继续递归处理下一层
                _set_weight_map_recursive(submodule)

        # 从根模块开始递归
        _set_weight_map_recursive(self)

        result = func(self, *args, **kwargs)  # 调用原始函数并保存返回值

        # 如果返回值是可迭代的，比如列表或元组，对每个元素进行标记
        if isinstance(result, (list, tuple)):
            result = [_C.op.makeOutput(self.model_desc, item) for item in result]
        # 如果返回值是单个值，直接进行标记
        else:
            result = _C.op.makeOutput(self.model_desc, result)

        # 初始化Net
        self.net.setModelDesc(self.model_desc)
        self.net.setDeviceType(self.device_type)
        self.net.enableOpt(kwargs.get("enable_net_opt", True))
        # 从kwargs中获取enable_pass，如果不存在则默认为一个空集合

        enable_pass = kwargs.get("enable_pass", set())
        # 从kwargs中获取disable_pass，如果不存在则默认为一个空集合
        disable_pass = kwargs.get("disable_pass", set())

        # 如果传入的不是集合，尝试将它们转换为集合
        if not isinstance(enable_pass, set):
            enable_pass = set(enable_pass)
        if not isinstance(disable_pass, set):
            disable_pass = set(disable_pass)

        self.net.setEnablePass(enable_pass)
        self.net.setDisablePass(disable_pass)
        if self.weight_map is not None:
            self.model_desc.set_weights(self.weight_map)
        else:
            warnings.warn(
                "Warning: weight_map is not set! Model weights are not initialized.",
                UserWarning,
                stacklevel=2,
            )
        self.net.init()

        return result

    return wrapper


def forward(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        # 递归地给所有子模块（含多级嵌套）设置 weight_map
        def _set_weight_map_recursive(module):
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue  # 跳过私有/保护成员
                submodule = getattr(module, attr_name, None)
                if not isinstance(submodule, Module):
                    continue  # 不是 Module，跳过

                # 如果当前子模块自己有 weight_map，就赋值
                if hasattr(submodule, "weight_map"):
                    sub_wm = getattr(submodule, "weight_map")
                    if sub_wm is not None:
                        for k in sub_wm:
                            if k not in self.weight_map:
                                raise KeyError(
                                    f"weight '{k}' is not initialized in Model.weight_map"
                                )
                            sub_wm[k] = self.weight_map[k]

                # 继续递归处理下一层
                _set_weight_map_recursive(submodule)

        # 从根模块开始递归
        _set_weight_map_recursive(self)

        result = func(self, *args, **kwargs)  # 调用原始函数并保存返回值

        return result

    return wrapper


class Module:

    def __init__(self):
        self.model_desc = ModelDesc()
        self.net = Net()
        self.device_type = DeviceType("cpu", 0)
        self.weight_map = None

    @build_model
    def construct(self, enable_net_opt=True, enable_pass=set(), disable_pass=set()):
        raise NotImplementedError()

    def run(self) -> list:
        self.net.preRun()
        self.net.run()
        self.net.postRun()
        return self.net.getAllOutput()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def makeExpr(self, *args, **kwargs):
        raise NotImplementedError()

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
