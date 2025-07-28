# cli
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

./nndeploy_demo_segment_anything --name nndeploy::segment_anything::SegmentAnythingGraph --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value ../../model/segment_anything/SAM_encoder.onnx,../../model/segment_anything/SAM_mask_decoder.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path ../../docs/image/demo/segment/sample.jpg --output_path ../../docs/image/demo/segment/sample_segment_anything.jpg --point_label 1 --points 100,100


./nndeploy_demo_segment_anything --name nndeploy::segment_anything::SegmentAnythingGraph --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value D:\Program\xnyh\nndeploy\model\segment_anything\sam_vit_b_01ec64_encoder_name.onnx,D:\Program\xnyh\nndeploy\model\segment_anything\sam_vit_b_01ec64_decoder.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path D:\Program\xnyh\nndeploy\docs\image\demo\segment\sample.jpg --output_path D:\Program\xnyh\nndeploy\docs\image\demo\segment\sample_sam.jpg --point_label 1 --points 400 700
```