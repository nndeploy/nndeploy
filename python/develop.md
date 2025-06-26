# python

## build & install

1. 开启编译选项

+ 确保在根目录使用`git submodule init && git submodule update`下载了所有子模块，如果拉取子模块失败，调用克隆子模块脚本
python3 clone_submodule.py

+ 在`build/config.cmake`将`ENABLE_NNDEPLOY_PYTHON`设置为`ON`

+ 参照`nndeploy`的构建文档构建完成，执行`make install`

注：修改了cpp代码，都需要重新`make install`

2. 安装Python包

```bash
cd nndeploy/python
pip install -e .  # 开发者模式安装
```
