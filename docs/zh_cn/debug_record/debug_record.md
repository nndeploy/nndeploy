# 开发调试问题记录

## Python

python binding相关，移步`python/README.md`

## 使用safetensors加载的权重修改报错

safetensors方式加载的权重直接mmap到文件，以只读方式打开，无法修改。如果要修改权重，clone一份Tensor修改，并替换。

## Ascend开发环境

### 编译报错，链接错误，undefined reference

需要使用root方式编译

### Vscode终端无路径提示、部分常用命令错误、无法输入等

Vscode默认打开的终端有残缺，需要手动新建终端，选择Bash