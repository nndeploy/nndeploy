from nndeploy.op import Module
from nndeploy.ir import ModelDesc
from nndeploy.base import DeviceType

import nndeploy._nndeploy_internal as _C

from .net import Net

from functools import wraps


def build_model(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        weight_maps = []
        module_count = 0

        for attr_name in dir(self):
            submodule = getattr(self, attr_name)

            if isinstance(submodule, Module):
                submodule.model_desc = self.model_desc  # 给所有的Op类型设置model_desc
                # if self.weight_map == None:
                #     if hasattr(submodule, "generateWeight"):
                #         generateWeight = getattr(submodule, "generateWeight")
                #         weight_map = generateWeight()
                #         # 为每个键添加前缀
                #         # 遍历列表中的每个字典
                #         for dic in weight_map:
                #             # 遍历字典的每个键
                #             for key in list(dic.keys()):  # 使用list(dic.keys())来避免在遍历时修改字典
                #                 # 构造新的键名
                #                 new_key = f"{module_count}_{key}"
                #                 # 将原键的值赋给新键，并删除原键
                #                 dic[new_key] = dic.pop(key)
                #         submodule.weight_map = weight_map
                #         weight_maps.append(weight_map)
                #         module_count += 1

                #     # 向ModelDesc设置权重
                if hasattr(submodule, "weight_map"):
                    # print(submodule)
                    module_weight_map = getattr(submodule, "weight_map")
                    if module_weight_map != None:
                        for k, v in module_weight_map.items():
                            if k not in self.weight_map:
                                raise KeyError(
                                    f"weight '{k}' is not initialized in Model.weight_map"
                                )
                            module_weight_map[k] = self.weight_map[k]

        # 对输入执行MakeInput标记

        result = func(self, *args, **kwargs)  # 调用原始函数并保存返回值

        # 如果返回值是可迭代的，比如列表或元组，对每个元素进行标记
        if isinstance(result, (list, tuple)):
            result = [_C.op.makeOutput(self.model_desc, item) for item in result]
        # 如果返回值是单个值，直接进行标记
        else:
            result = _C.op.makeOutput(self.model_desc, result)

        # for weight_map in weight_maps:
        #     for item in weight_map:
        #         for name, tensor in item.items():
        #             self.model_desc.weights_[name] = tensor

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

        self.model_desc.setWeights(self.weight_map)
        self.net.init()

        return result

    return wrapper


class Model:

    def __init__(self):
        self.model_desc = ModelDesc()
        self.net = Net()
        self.device_type = DeviceType("cpu", 0)
        self.weight_map = None  # 如果判定weight_map是空的，那就随机生成名字和权重数据

    @build_model
    def construct(self, enable_net_opt=True, enable_pass=set(), disable_pass=set()):
        raise NotImplementedError()

    def run(self):
        self.net.preRun()
        self.net.run()
        self.net.postRun()
        return self.net.getAllOutput()
