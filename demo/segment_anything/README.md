# 分割 (Segment)

图像分割演示程序，支持 SAM、SAM2、RMBG、YOLO-Seg。

## 支持的算法

| 算法 | 工作流 JSON | 说明 |
|------|-------------|------|
| SAM | `workflow/segment/Segment_Anything.json` | SAM v1/vit-b/vit-h，提示点/框分割 |
| SAM2 | `workflow/segment/Segment_SAM2.json` | SAM2 多节点工作流（编码器+点/掩码/解码器） |
| RMBG | `workflow/segment/Segment_RMBG.json` | 背景去除 |
| YOLO-Seg | `workflow/segment/Segment_YOLO-Seg.json` | YOLO实例分割 |

## 运行 JSON 工作流

```bash
cd path/to/nndeploy

# SAM
./nndeploy_demo_segment_anything --json_file resources/workflow/segment/Segment_Anything.json

# SAM2
./nndeploy_demo_segment_anything --json_file resources/workflow/segment/Segment_SAM2.json

# YOLO-Seg
./nndeploy_demo_segment_anything --json_file resources/workflow/segment/Segment_YOLO-Seg.json

# RMBG
./nndeploy_demo_segment_anything --json_file resources/workflow/segment/Segment_RMBG.json
```

## CLI 命令行（无 JSON）

```shell
./nndeploy_demo_segment_anything --name nndeploy::segment_anything::SegmentAnythingGraph \
    --inference_type kInferenceTypeOnnxRuntime \
    --device_type kDeviceTypeCodeX86:0 \
    --model_type kModelTypeOnnx \
    --is_path \
    --model_value ../../model/segment_anything/SAM_encoder.onnx,../../model/segment_anything/SAM_mask_decoder.onnx \
    --codec_flag kCodecFlagImage \
    --parallel_type kParallelTypeSequential \
    --input_path ../../docs/image/demo/segment/sample.jpg \
    --output_path ../../docs/image/demo/segment/sample_segment_anything.jpg \
    --point_label 1 \
    --points 100,100,200,200
```