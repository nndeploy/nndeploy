# Segment Anything — 实现目录

本目录包含 SAM / SAM 2 / SAM 3 的 C++ 实现源文件。

| 文件 | 对应版本 | 行数 | 说明 |
|------|---------|------|------|
| `sam.cc` | SAM (v1) | ~320 | SelectPointNode + SAMGraph 预处理/后处理/推理连线 |
| `sam2.cc` | SAM 2 | ~470 | SAM2PointNode + SAM2MaskNode + SAM2PostProcess + SAM2MemoryNode |
| `sam3.cc` | SAM 3 | ~2930 | Legacy 12+ 节点 + Simplified 3 节点的全部实现 |

详细文档请参阅头文件目录下的 README：
[../../include/nndeploy/segment/segment_anything/README.md](../../include/nndeploy/segment/segment_anything/README.md)
