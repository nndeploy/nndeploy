# 搭建Ascend环境

## 硬件与软件要求
| 类别     | 版本                 | 说明       |
|--------|--------------------|----------|
| Python | Python3.10         | 面向对象编程语言 |
| CANN   | 8.0.RC3.alpha003      | 昇腾异构计算架构 |
| 系统镜像   | Ubuntu 22.04.5 LTS | 服务器操作系统  |
| CPU架构  | aarch64            | CPU架构    |
| NPU    | Atlas 800 A2               | NPU型号    |

## 安装包准备
[CANN社区版下载网址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.alpha003)\
下载好以下两个安装包\
Ascend-cann-toolkit_8.0.RC3.alpha003_linux-aarch64.run\
Ascend-cann-kernels-910b_8.0.RC3.alpha003_linux-aarch64.run

## 环境安装
参考[官方文档](https://www.hiascend.com/document/detail/zh/canncommercial/700/quickstart/quickstart/quickstart_18_0002.html)\
安装CANN
```
sudo chmod +x Ascend-cann-toolkit_8.0.RC3.alpha003_linux-aarch64.run
./Ascend-cann-toolkit_8.0.RC3.alpha003_linux-aarch64.run --install --force 
```
安装Kernel
```
sudo chmod +x Ascend-cann-kernels-910b_8.0.RC3.alpha003_linux-aarch64.run 
sudo ./Ascend-cann-kernels-910b_8.0.RC3.alpha003_linux-aarch64.run --install --install-for-all
```

更新环境 (如若指定了安装路径，请对应的修改此处的path)
```
sudo echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /etc/profile
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

验证环境是否安装成功
```
git clone https://gitee.com/ascend/samples.git
cd samples/operator/AddCustomSample/KernelLaunch/AddKernelInvocation
bash run.sh -v Ascend910B4 -r npu
```
若成功则会如下图显示
![Alt text](../../image/kernel%20sample.png)