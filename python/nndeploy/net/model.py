from nndeploy.op import Module
from nndeploy.ir import ModelDesc

import nndeploy._nndeploy_internal as _C



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
                if self.weight_file == None:
                    if hasattr(submodule, "generateWeight"):
                        generateWeight = getattr(submodule, "generateWeight")
                        weight_map = generateWeight()
                        # 为每个键添加前缀
                        # 遍历列表中的每个字典
                        for dic in weight_map:
                            # 遍历字典的每个键
                            for key in list(dic.keys()):  # 使用list(dic.keys())来避免在遍历时修改字典
                                # 构造新的键名
                                new_key = f"{module_count}_{key}"
                                # 将原键的值赋给新键，并删除原键
                                dic[new_key] = dic.pop(key)
                        submodule.weight_map = weight_map
                        weight_maps.append(weight_map)
                        module_count += 1

        # 对输入执行MakeInput标记

        result = func(self, *args, **kwargs)  # 调用原始函数并保存返回值

        # 如果返回值是可迭代的，比如列表或元组，对每个元素进行标记
        if isinstance(result, (list, tuple)):
            result = [_C.op.makeOutput(self.model_desc,item) for item in result]
        # 如果返回值是单个值，直接进行标记
        else:
            result = _C.op.makeOutput(self.model_desc,result)

        for weight_map in weight_maps:
            for item in weight_map:
                for name, tensor in item.items():
                    self.model_desc.weights_[name] = tensor

        return result

    return wrapper


class Model:

    def __init__(self):
        self.model_desc = ModelDesc()
        self.weight_file = None  # 如果判定weight_file是空的，那就随机生成名字和权重数据

    @build_model
    def construct():
        raise NotImplementedError()
