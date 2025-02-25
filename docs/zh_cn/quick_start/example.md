# 示例工程

## 从源码编译

参考编译文档，[编译文档链接](./build.md)

## 基于DAG的模型部署演示示例（采用默认config.cmake即可编译成功）

### Windows 下运行 nndeploy_demo_dag
```shell
cd /yourpath/nndeploy/build/install/bin
.\nndeploy_demo_dag.exe
```

### Linux 下运行 nndeploy_demo_dag
```shell
cd /yourpath/nndeploy/build/install/lib
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
./nndeploy_demo_dag
```

### Andorid 下运行 nndeploy_demo_dag
```shell
cd /yourpath/nndeploy/build/install/lib

adb push * /data/local/tmp/

adb shell 

cd /data/local/tmp/

export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
./nndeploy_demo_dag
```

### 效果示例
```shell
E/nndeploy_default_str: main [File //yourpath/nndeploy/demo/dag/demo.cc][Line 273] start!
digraph serial_graph {
p0x7ffeec419690[shape=box, label=graph_in]
p0x7ffeec419690->p0x5614d9427700[label=graph_in]
p0x5614d9427700[label=model_0_graph]
p0x5614d9427700->p0x5614d941c230[label=model_0_out]
p0x5614d941c230[label=op_link]
p0x5614d941c230->p0x5614d9428260[label=op_link_out]
p0x5614d9428260[label=model_1_graph]
p0x7ffeec4196c0[shape=box, label=graph_out]
p0x5614d9428260->p0x7ffeec4196c0[label=graph_out]
}
digraph model_0_graph {
p0x7ffeec419690[shape=box, label=graph_in]
p0x7ffeec419690->p0x5614d9420e00[label=graph_in]
p0x5614d9420e00[label=model_0_graph_preprocess]
p0x5614d9420e00->p0x5614d9418e00[label=model_0_graph_preprocess_out]
p0x5614d9418e00[label=model_0_graph_infer]
p0x5614d9418e00->p0x5614d941b100[label=model_0_graph_infer_out]
p0x5614d941b100[label=model_0_graph_postprocess]
p0x5614d94275a0[shape=box, label=model_0_out]
p0x5614d941b100->p0x5614d94275a0[label=model_0_out]
}
digraph model_1_graph {
p0x5614d9427650[shape=box, label=op_link_out]
p0x5614d9427650->p0x5614d9419b60[label=op_link_out]
p0x5614d9419b60[label=model_1_graph_preprocess]
p0x5614d9419b60->p0x5614d94c0b00[label=model_1_graph_preprocess_out]
p0x5614d94c0b00[label=model_1_graph_infer]
p0x5614d94c0b00->p0x5614d94c0d80[label=model_1_graph_infer_out]
p0x5614d94c0d80[label=model_1_graph_postprocess]
p0x7ffeec4196c0[shape=box, label=graph_out]
p0x5614d94c0d80->p0x7ffeec4196c0[label=graph_out]
}
I/nndeploy_default_str: run [File /home/always/github/public/nndeploy/demo/dag/demo.cc][Line 46] running node = [model_0_graph_preprocess]!
I/nndeploy_default_str: run [File /home/always/github/public/nndeploy/demo/dag/demo.cc][Line 46] running node = [model_0_graph_infer]!
I/nndeploy_default_str: run [File /home/always/github/public/nndeploy/demo/dag/demo.cc][Line 46] running node = [model_0_graph_postprocess]!
I/nndeploy_default_str: run [File /home/always/github/public/nndeploy/demo/dag/demo.cc][Line 46] running node = [op_link]!
I/nndeploy_default_str: run [File /home/always/github/public/nndeploy/demo/dag/demo.cc][Line 46] running node = [model_1_graph_preprocess]!
I/nndeploy_default_str: run [File /home/always/github/public/nndeploy/demo/dag/demo.cc][Line 46] running node = [model_1_graph_infer]!
I/nndeploy_default_str: run [File /home/always/github/public/nndeploy/demo/dag/demo.cc][Line 46] running node = [model_1_graph_postprocess]!
E/nndeploy_default_str: main [File /home/always/github/public/nndeploy/demo/dag/demo.cc][Line 350] end!
```
