import Mock from "mockjs";
import {
  INodeBranchEntity,
  INodeEntity,
  INodeTreeNodeEntity,
} from "../../../pages/Node/entity";
import { MockItem } from "../../entity";

// mock方法,详细的可以看官方文档
const Random = Mock.Random;

const nodeBranches: INodeBranchEntity[] = [
  {
    id: "1",
    name: "begin",
    parentId: "",
  },
  {
    id: "2",
    name: "llm",
    parentId: "",
  },
  {
    id: "3",
    name: "branch",
    parentId: "",
  },
  {
    id: "3-1",
    name: "loop",
    parentId: "3",
  },
  {
    id: "3-2",
    name: "condition",
    parentId: "3",
  },
  {
    id: "4",
    name: "end",
    parentId: "",
  },
];

// const nodes: INodeEntity[] = [
//         {
//             "key_": "nndeploy::preprocess::CvtColorResize",
//             "name_": "preprocess",
//             "device_type_": "kDeviceTypeCodeX86:0",
//             "inputs_": [
//                 {
//                     "desc_": "detect_in",
//                     "type_": "Mat"
//                 },
//                 //  {
//                 //     "desc_": "detect_in2",
//                 //     "type_": "Mat"
//                 // }
//             ],
//             "outputs_": [
//                 {
//                     "desc_": "images",
//                     "type_": "Tensor"
//                 }
//             ],
//             "param_": {
//                 "src_pixel_type_": "kPixelTypeBGR",
//                 "dst_pixel_type_": "kPixelTypeRGB",
//                 "interp_type_": "kInterpTypeLinear",
//                 "h_": 640,
//                 "w_": 640,
//                 "data_type_": "kDataTypeCodeFp 32 1",
//                 "data_format_": "kDataFormatNCHW",
//                 "normalize_": true,
//                 "scale_": [
//                     0.003921568859368563,
//                     0.003921568859368563,
//                     0.003921568859368563,
//                     0.003921568859368563
//                 ],
//                 "mean_": [
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0
//                 ],
//                 "std_": [
//                     1.0,
//                     1.0,
//                     1.0,
//                     1.0
//                 ]
//             }
//         },
//         {
//             "key_": "nndeploy::infer::Infer",
//             "name_": "infer",
//             "device_type_": "kDeviceTypeCodeX86:0",
//             "inputs_": [
//                 {
//                     "desc_": "input0",
//                     "type_": "Tensor"
//                 }
//             ],
//             "outputs_": [
//                 {
//                     "desc_": "output0",
//                     "type_": "Tensor"
//                 }
//             ],
//             is_dynamic_input_: true,
//             is_dynamic_output_: true,
//             "type_": "kInferenceTypeOnnxRuntime",
//             "param_": {
//                 "inference_type_": "kInferenceTypeOnnxRuntime",
//                 "model_type_": "kModelTypeOnnx",
//                 "is_path_": true,
//                 "model_value_": [
//                     "yolo11s.sim.onnx"
//                 ],
//                 "input_num_": 0,
//                 "input_name_": [],
//                 "input_shape_": [],
//                 "output_num_": 0,
//                 "output_name_": [],
//                 "encrypt_type_": "kEncryptTypeNone",
//                 "license_": "",
//                 "device_type_": "kDeviceTypeCodeX86:0",
//                 "num_thread_": 4,
//                 "gpu_tune_kernel_": 1,
//                 "share_memory_mode_": "kShareMemoryTypeNoShare",
//                 "precision_type_": "kPrecisionTypeFp32",
//                 "power_type_": "kPowerTypeNormal",
//                 "is_dynamic_shape_": false,
//                 "min_shape_": {},
//                 "opt_shape_": {},
//                 "max_shape_": {},
//                 "parallel_type_": "kParallelTypeSequential",
//                 "worker_num_": 4
//             }
//         },
//         {
//             "key_": "nndeploy::detect::YoloPostProcess",
//             "name_": "postprocess",
//             "device_type_": "kDeviceTypeCodeX86:0",
//             "inputs_": [
//                 {
//                     "desc_": "output0",
//                     "type_": "Tensor"
//                 }
//             ],
//             "outputs_": [
//                 {
//                     "desc_": "detect_out",
//                     "type_": "Param"
//                 }
//             ],
//             "param_": {
//                 "version_": 11,
//                 "score_threshold_": 0.5,
//                 "nms_threshold_": 0.44999998807907104,
//                 "num_classes_": 80,
//                 "model_h_": 640,
//                 "model_w_": 640
//             }
//         },
//         {
//             "key_": "nndeploy::detect::YoloGraph",
//             "name_": "nndeploy::detect::YoloGraph",
//             "device_type_": "kDeviceTypeCodeX86:0:0",
//             is_graph_: true,
//             "inputs_": [
//                 {
//                     "desc_": "detect_in",
//                     "type_": "CvMat"
//                 }
//             ],
//             "outputs_": [
//                 {
//                     "desc_": "detect_out",
//                     "type_": "Param"
//                 }
//             ],
//             "param_": {

//             }

//         },
//         {
//             "key_": "nndeploy::codec::OpenCvImageDecodeNode",
//             "name_": "decode_node",
//             "device_type_": "kDeviceTypeCodeCpu:0",
//             "inputs_": [],
//             "outputs_": [
//                 {
//                     "desc_": "detect_in",
//                     "type_": "CvMat"
//                 }
//             ],
//             "flag_": "kCodecFlagImage",
//             "param_": {

//             }
//         },
//         {
//             "key_": "nndeploy::detect::DrawBoxNode",
//             "name_": "DrawBoxNode",
//             "device_type_": "kDeviceTypeCodeX86:0",
//             "inputs_": [
//                 {
//                     "desc_": "detect_in",
//                     "type_": "CvMat"
//                 },
//                 {
//                     "desc_": "detect_out",
//                     "type_": "Param"
//                 }
//             ],
//             "outputs_": [
//                 {
//                     "desc_": "draw_output",
//                     "type_": "CvMat"
//                 }
//             ],
//              "param_": {

//             }
//         },
//         {
//             "key_": "nndeploy::codec::OpenCvImageEncodeNode",
//             "name_": "encode_node",
//             "device_type_": "kDeviceTypeCodeCpu:0",
//             "inputs_": [
//                 {
//                     "desc_": "draw_output",
//                     "type_": "CvMat"
//                 }
//             ],
//             "outputs_": [],
//             "flag_": "kCodecFlagImage",
//             param_: {}
//         }
//     ]

const nodes: INodeEntity[] = [
  {
    key_: "nndeploy::preprocess::CvtResizeNormTransCropNormTrans",
    name_: "CvtResizeNormTransCropNormTrans",
    desc_:
      "cv::Mat to device::Tensor[cvtcolor->resize->crop->normalize->transpose]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Tensor",
        desc_: "output_0",
      },
    ],
    param_: {
      src_pixel_type_: "kPixelTypeGRAY",
      dst_pixel_type_: "kPixelTypeGRAY",
      interp_type_: "kInterpTypeNearst",
      data_type_: "kDataTypeCodeFp 32 1",
      data_format_: "kDataFormatNCHW",
      resize_h_: -1,
      resize_w_: -1,
      normalize_: true,
      scale_: [
        0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
        0.003921568859368563,
      ],
      mean_: [0.0, 0.0, 0.0, 0.0],
      std_: [1.0, 1.0, 1.0, 1.0],
      top_left_x_: 0,
      top_left_y_: 0,
      width_: 0,
      height_: 0,
    },
  },
  {
    key_: "nndeploy::codec::OpenCvVedioDecode",
    name_: "OpenCvVedioDecode",
    desc_:
      "Decode video using OpenCV, from video file to cv::Mat frames, default color space is BGR",
    device_type_: "kDeviceTypeCodeCpu:0",
    is_dynamic_input_: false,
    inputs_: [],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Mat",
        desc_: "output_0",
      },
    ],
    flag_: "kCodecFlagImage",
    path_: "",
    size_: 0,
  },
  {
    key_: "nndeploy::preprocess::CvtResizeNormTrans",
    name_: "CvtResizeNormTrans",
    desc_: "cv::Mat to device::Tensor[cvtcolor->resize->normalize->transpose]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Tensor",
        desc_: "output_0",
      },
    ],
    param_: {
      src_pixel_type_: "kPixelTypeGRAY",
      dst_pixel_type_: "kPixelTypeGRAY",
      interp_type_: "kInterpTypeNearst",
      h_: -1,
      w_: -1,
      data_type_: "kDataTypeCodeFp 32 1",
      data_format_: "kDataFormatNCHW",
      normalize_: true,
      scale_: [
        0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
        0.003921568859368563,
      ],
      mean_: [0.0, 0.0, 0.0, 0.0],
      std_: [1.0, 1.0, 1.0, 1.0],
    },
  },
  {
    key_: "nndeploy::segment::DrawMask",
    name_: "DrawMask",
    desc_:
      "Draw segmentation mask on input cv::Mat image based on segmentation results[cv::Mat->cv::Mat]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
      {
        type_: "Param",
        desc_: "input_1",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Mat",
        desc_: "output_0",
      },
    ],
  },
  {
    key_: "nndeploy::detect::YoloXPostProcess",
    name_: "YoloXPostProcess",
    desc_: "YOLOX postprocess[device::Tensor->DetectResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Tensor",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    param_: {
      score_threshold_: 0.0,
      nms_threshold_: 0.0,
      num_classes_: 0,
      model_h_: 0,
      model_w_: 0,
    },
  },
  {
    key_: "nndeploy::segment::RMBGPostProcess",
    name_: "RMBGPostProcess",
    desc_: "Segment RMBG postprocess[device::Tensor->SegmentResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Tensor",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    param_: {
      version: -1,
    },
  },
  {
    key_: "nndeploy::tokenizer::TokenizerEncodeCpp",
    name_: "TokenizerEncodeCpp",
    desc_:
      "A tokenizer encode node that uses the C++ tokenizers library to encode text into token IDs. Supports HuggingFace and BPE tokenizers. Can encode single strings or batches of text. Provides vocabulary lookup and token-to-ID conversion.",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Param",
        desc_: "input_0",
      },
      {
        type_: "Param",
        desc_: "input_1",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
      {
        type_: "Param",
        desc_: "output_1",
      },
    ],
    param_: {
      is_path_: true,
      tokenizer_type_: "kTokenizerTypeHF",
      json_blob_: "",
      model_blob_: "",
      vocab_blob_: "",
      merges_blob_: "",
      added_tokens_: "",
      max_length_: 77,
    },
  },
  {
    key_: "nndeploy::preprocess::WarpAffineCvtNormTrans",
    name_: "WarpAffineCvtNormTrans",
    desc_:
      "cv::Mat to device::Tensor[warpaffine->cvtcolor->normalize->transpose]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Tensor",
        desc_: "output_0",
      },
    ],
    param_: {
      transform_: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      dst_w_: 0,
      dst_h_: 0,
      src_pixel_type_: "kPixelTypeGRAY",
      dst_pixel_type_: "kPixelTypeGRAY",
      data_type_: "kDataTypeCodeFp 32 1",
      data_format_: "kDataFormatNCHW",
      h_: -1,
      w_: -1,
      normalize_: true,
      const_value_: 114,
      scale_: [
        0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
        0.003921568859368563,
      ],
      mean_: [0.0, 0.0, 0.0, 0.0],
      std_: [1.0, 1.0, 1.0, 1.0],
      interp_type_: "kInterpTypeLinear",
      border_type_: "kBorderTypeConstant",
      border_val_: [0.0, 0.0, 0.0, 0.0],
    },
  },
  {
    key_: "nndeploy::codec::OpenCvImageDecode",
    name_: "OpenCvImageDecode",
    desc_:
      "Decode image using OpenCV, from image path to cv::Mat, default color space is BGR",
    device_type_: "kDeviceTypeCodeCpu:0",
    is_dynamic_input_: false,
    inputs_: [],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Mat",
        desc_: "output_0",
      },
    ],
    flag_: "kCodecFlagImage",
    path_: "",
    size_: 0,
  },
  {
    key_: "nndeploy::codec::OpenCvVedioEncode",
    name_: "OpenCvVedioEncode",
    desc_:
      "Encode video using OpenCV, from cv::Mat frames to video file, supports common video formats",
    device_type_: "kDeviceTypeCodeCpu:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [],
    flag_: "kCodecFlagImage",
    path_: "",
    ref_path_: "",
    fourcc_: "mp4v",
    fps_: 0.0,
    width_: 0,
    height_: 0,
    size_: 0,
  },
  {
    key_: "nndeploy::infer::Infer",
    name_: "Infer",
    desc_: "",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: true,
    inputs_: [
      {
        type_: "Tensor",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: true,
    outputs_: [
      {
        type_: "Tensor",
        desc_: "output_0",
      },
    ],
    type_: "kInferenceTypeNotSupport",
    param_: {
      inference_type_: "kInferenceTypeNone",
      model_type_: "kModelTypeNotSupport",
      is_path_: false,
      input_num_: 0,
      input_name_: [],
      input_shape_: [],
      output_num_: 0,
      output_name_: [],
      encrypt_type_: "kEncryptTypeNone",
      license_: "",
      device_type_: "kDeviceTypeCodeCpu:0",
      num_thread_: 1,
      gpu_tune_kernel_: 1,
      share_memory_mode_: "kShareMemoryTypeNoShare",
      precision_type_: "kPrecisionTypeFp32",
      power_type_: "kPowerTypeNormal",
      is_dynamic_shape_: false,
      min_shape_: {},
      opt_shape_: {},
      max_shape_: {},
      parallel_type_: "kParallelTypeNone",
      worker_num_: 4,
    },
  },
  {
    key_: "nndeploy::matting::PPMattingGraph",
    name_: "PPMattingGraph",
    desc_:
      "PPMatting graph[cv::Mat->preprocess->infer->postprocess->MattingResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    is_graph_: true,
    parallel_type_: "kParallelTypeNone",
    is_inner_: false,
    is_time_profile_: false,
    is_debug_: false,
    is_external_stream_: false,
    is_graph_node_share_stream_: true,
    queue_max_size_: 16,
    node_repository_: [
      {
        key_: "nndeploy::preprocess::CvtResizePadNormTrans",
        name_: "preprocess",
        desc_: "cv::Mat to device::Tensor[resize->pad->normalize->transpose]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Mat",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        param_: {
          src_pixel_type_: "kPixelTypeGRAY",
          dst_pixel_type_: "kPixelTypeGRAY",
          interp_type_: "kInterpTypeNearst",
          data_type_: "kDataTypeCodeFp 32 1",
          data_format_: "kDataFormatNCHW",
          h_: -1,
          w_: -1,
          normalize_: true,
          scale_: [
            0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
            0.003921568859368563,
          ],
          mean_: [0.0, 0.0, 0.0, 0.0],
          std_: [1.0, 1.0, 1.0, 1.0],
          border_type_: "kBorderTypeConstant",
          top_: 0,
          bottom_: 0,
          left_: 0,
          right_: 0,
          border_val_: [0.0, 0.0, 0.0, 0.0],
        },
      },
      {
        key_: "nndeploy::infer::Infer",
        name_: "infer",
        desc_: "",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: true,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: true,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        type_: "kInferenceTypeNotSupport",
        param_: {
          inference_type_: "kInferenceTypeNone",
          model_type_: "kModelTypeDefault",
          is_path_: false,
          input_num_: 0,
          input_name_: [],
          input_shape_: [],
          output_num_: 0,
          output_name_: [],
          encrypt_type_: "kEncryptTypeNone",
          license_: "",
          device_type_: "kDeviceTypeCodeCpu:0",
          num_thread_: 1,
          gpu_tune_kernel_: 1,
          share_memory_mode_: "kShareMemoryTypeNoShare",
          precision_type_: "kPrecisionTypeFp32",
          power_type_: "kPowerTypeNormal",
          is_dynamic_shape_: false,
          min_shape_: {},
          opt_shape_: {},
          max_shape_: {},
          parallel_type_: "kParallelTypeNone",
          worker_num_: 4,
        },
      },
      {
        key_: "nndeploy::matting::PPMattingPostProcess",
        name_: "postprocess",
        desc_: "Matting postprocess[device::Tensor->MattingResult]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Param",
            desc_: "output_0",
          },
        ],
        param_: {
          alpha_h: 0,
          alpha_w: 0,
          output_h: 0,
          output_w: 0,
        },
      },
    ],
  },
  {
    key_: "nndeploy::preprocess::ConvertTo",
    name_: "ConvertTo",
    desc_:
      "Convert the data type of the input tensor to the specified data type",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Tensor",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Tensor",
        desc_: "output_0",
      },
    ],
    param_: {
      dst_data_type_: "kDataTypeCodeFp 32 1",
    },
  },
  {
    key_: "nndeploy::detect::YoloMultiOutputPostProcess",
    name_: "YoloMultiOutputPostProcess",
    desc_: "YOLO multi-output postprocess[device::Tensor->DetectResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Tensor",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    param_: {
      version_: -1,
      score_threshold_: 0.0,
      nms_threshold_: 0.0,
      num_classes_: 0,
      model_h_: 0,
      model_w_: 0,
      anchors_stride_8: [10, 13, 16, 30, 33, 23],
      anchors_stride_16: [30, 61, 62, 45, 59, 119],
      anchors_stride_32: [116, 90, 156, 198, 373, 326],
    },
  },
  {
    key_: "nndeploy::classification::ClassificationPostProcess",
    name_: "ClassificationPostProcess",
    desc_: "Classification postprocess[device::Tensor->ClassificationResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Tensor",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    param_: {
      topk: 1,
      is_softmax: true,
      version: -1,
    },
  },
  {
    key_: "nndeploy::preprocess::CvtResizePadNormTrans",
    name_: "CvtResizePadNormTrans",
    desc_: "cv::Mat to device::Tensor[resize->pad->normalize->transpose]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Tensor",
        desc_: "output_0",
      },
    ],
    param_: {
      src_pixel_type_: "kPixelTypeGRAY",
      dst_pixel_type_: "kPixelTypeGRAY",
      interp_type_: "kInterpTypeNearst",
      data_type_: "kDataTypeCodeFp 32 1",
      data_format_: "kDataFormatNCHW",
      h_: -1,
      w_: -1,
      normalize_: true,
      scale_: [
        0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
        0.003921568859368563,
      ],
      mean_: [0.0, 0.0, 0.0, 0.0],
      std_: [1.0, 1.0, 1.0, 1.0],
      border_type_: "kBorderTypeConstant",
      top_: 0,
      bottom_: 0,
      left_: 0,
      right_: 0,
      border_val_: [0.0, 0.0, 0.0, 0.0],
    },
  },
  {
    key_: "nndeploy::detect::YoloPostProcess",
    name_: "YoloPostProcess",
    desc_: "YOLO v5/v6/v7/v8/v11 postprocess[device::Tensor->DetectResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Tensor",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    param_: {
      version_: -1,
      score_threshold_: 0.0,
      nms_threshold_: 0.0,
      num_classes_: 0,
      model_h_: 0,
      model_w_: 0,
    },
  },
  {
    key_: "nndeploy::detect::YoloMultiOutputGraph",
    name_: "YoloMultiOutputGraph",
    desc_:
      "yolo v5/v6/v7/v8/v11 graph[cv::Mat->preprocess->infer->postprocess->DetectResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    is_graph_: true,
    parallel_type_: "kParallelTypeNone",
    is_inner_: false,
    is_time_profile_: false,
    is_debug_: false,
    is_external_stream_: false,
    is_graph_node_share_stream_: true,
    queue_max_size_: 16,
    node_repository_: [
      {
        key_: "nndeploy::preprocess::CvtResizeNormTrans",
        name_: "preprocess",
        desc_:
          "cv::Mat to device::Tensor[cvtcolor->resize->normalize->transpose]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Mat",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        param_: {
          src_pixel_type_: "kPixelTypeGRAY",
          dst_pixel_type_: "kPixelTypeGRAY",
          interp_type_: "kInterpTypeNearst",
          h_: -1,
          w_: -1,
          data_type_: "kDataTypeCodeFp 32 1",
          data_format_: "kDataFormatNCHW",
          normalize_: true,
          scale_: [
            0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
            0.003921568859368563,
          ],
          mean_: [0.0, 0.0, 0.0, 0.0],
          std_: [1.0, 1.0, 1.0, 1.0],
        },
      },
      {
        key_: "nndeploy::infer::Infer",
        name_: "infer",
        desc_: "",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: true,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: true,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        type_: "kInferenceTypeNotSupport",
        param_: {
          inference_type_: "kInferenceTypeNone",
          model_type_: "kModelTypeDefault",
          is_path_: false,
          input_num_: 0,
          input_name_: [],
          input_shape_: [],
          output_num_: 0,
          output_name_: [],
          encrypt_type_: "kEncryptTypeNone",
          license_: "",
          device_type_: "kDeviceTypeCodeCpu:0",
          num_thread_: 1,
          gpu_tune_kernel_: 1,
          share_memory_mode_: "kShareMemoryTypeNoShare",
          precision_type_: "kPrecisionTypeFp32",
          power_type_: "kPowerTypeNormal",
          is_dynamic_shape_: false,
          min_shape_: {},
          opt_shape_: {},
          max_shape_: {},
          parallel_type_: "kParallelTypeNone",
          worker_num_: 4,
        },
      },
      {
        key_: "nndeploy::detect::YoloMultiOutputPostProcess",
        name_: "postprocess",
        desc_: "YOLO multi-output postprocess[device::Tensor->DetectResult]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Param",
            desc_: "output_0",
          },
        ],
        param_: {
          version_: -1,
          score_threshold_: 0.0,
          nms_threshold_: 0.0,
          num_classes_: 0,
          model_h_: 0,
          model_w_: 0,
          anchors_stride_8: [10, 13, 16, 30, 33, 23],
          anchors_stride_16: [30, 61, 62, 45, 59, 119],
          anchors_stride_32: [116, 90, 156, 198, 373, 326],
        },
      },
    ],
  },
  {
    key_: "nndeploy::preprocess::BatchPreprocess",
    name_: "BatchPreprocess",
    desc_:
      "std::vector<cv::Mat> to device::Tensor, support all preprocess nodes[cv::Mat->device::Tensor]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "St6vectorIN2cv3MatESaIS1_EE",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Tensor",
        desc_: "output_0",
      },
    ],
    data_format_: "kDataFormatNCHW",
    node_key_: "",
  },
  {
    key_: "nndeploy::classification::ClassificationGraph",
    name_: "ClassificationGraph",
    desc_:
      "Classification graph[cv::Mat->preprocess->infer->postprocess->ClassificationResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    is_graph_: true,
    parallel_type_: "kParallelTypeNone",
    is_inner_: false,
    is_time_profile_: false,
    is_debug_: false,
    is_external_stream_: false,
    is_graph_node_share_stream_: true,
    queue_max_size_: 16,
    node_repository_: [
      {
        key_: "nndeploy::preprocess::CvtResizeNormTransCropNormTrans",
        name_: "preprocess",
        desc_:
          "cv::Mat to device::Tensor[cvtcolor->resize->crop->normalize->transpose]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Mat",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        param_: {
          src_pixel_type_: "kPixelTypeGRAY",
          dst_pixel_type_: "kPixelTypeGRAY",
          interp_type_: "kInterpTypeNearst",
          data_type_: "kDataTypeCodeFp 32 1",
          data_format_: "kDataFormatNCHW",
          resize_h_: -1,
          resize_w_: -1,
          normalize_: true,
          scale_: [
            0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
            0.003921568859368563,
          ],
          mean_: [0.0, 0.0, 0.0, 0.0],
          std_: [1.0, 1.0, 1.0, 1.0],
          top_left_x_: 0,
          top_left_y_: 0,
          width_: 0,
          height_: 0,
        },
      },
      {
        key_: "nndeploy::infer::Infer",
        name_: "infer",
        desc_: "",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: true,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: true,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        type_: "kInferenceTypeNotSupport",
        param_: {
          inference_type_: "kInferenceTypeNone",
          model_type_: "kModelTypeDefault",
          is_path_: false,
          input_num_: 0,
          input_name_: [],
          input_shape_: [],
          output_num_: 0,
          output_name_: [],
          encrypt_type_: "kEncryptTypeNone",
          license_: "",
          device_type_: "kDeviceTypeCodeCpu:0",
          num_thread_: 1,
          gpu_tune_kernel_: 1,
          share_memory_mode_: "kShareMemoryTypeNoShare",
          precision_type_: "kPrecisionTypeFp32",
          power_type_: "kPowerTypeNormal",
          is_dynamic_shape_: false,
          min_shape_: {},
          opt_shape_: {},
          max_shape_: {},
          parallel_type_: "kParallelTypeNone",
          worker_num_: 4,
        },
      },
      {
        key_: "nndeploy::classification::ClassificationPostProcess",
        name_: "postprocess",
        desc_:
          "Classification postprocess[device::Tensor->ClassificationResult]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Param",
            desc_: "output_0",
          },
        ],
        param_: {
          topk: 1,
          is_softmax: true,
          version: -1,
        },
      },
    ],
  },
  {
    key_: "nndeploy::preprocess::CvtNormTrans",
    name_: "CvtNormTrans",
    desc_: "cv::Mat to device::Tensor[cvtcolor->normalize->transpose]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Tensor",
        desc_: "output_0",
      },
    ],
    param_: {
      src_pixel_type_: "kPixelTypeGRAY",
      dst_pixel_type_: "kPixelTypeGRAY",
      data_type_: "kDataTypeCodeFp 32 1",
      data_format_: "kDataFormatNCHW",
      normalize_: true,
      scale_: [
        0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
        0.003921568859368563,
      ],
      mean_: [0.0, 0.0, 0.0, 0.0],
      std_: [1.0, 1.0, 1.0, 1.0],
    },
  },
  {
    key_: "nndeploy::codec::BatchOpenCvDecode",
    name_: "BatchOpenCvDecode",
    desc_:
      "BatchOpenCvDecode node for decoding batches of images/videos using OpenCV",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "St6vectorIN2cv3MatESaIS1_EE",
        desc_: "output_0",
      },
    ],
    batch_size_: 1,
    node_key_: "",
  },
  {
    key_: "nndeploy::track::FairMotPreProcess",
    name_: "FairMotPreProcess",
    desc_: "FairMot preprocess[cv::Mat->device::Tensor]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Tensor",
        desc_: "output_0",
      },
      {
        type_: "Tensor",
        desc_: "output_1",
      },
      {
        type_: "Tensor",
        desc_: "output_2",
      },
    ],
    param_: {
      src_pixel_type_: "kPixelTypeGRAY",
      dst_pixel_type_: "kPixelTypeGRAY",
      interp_type_: "kInterpTypeNearst",
      h_: -1,
      w_: -1,
      data_type_: "kDataTypeCodeFp 32 1",
      data_format_: "kDataFormatNCHW",
      normalize_: true,
      scale_: [
        0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
        0.003921568859368563,
      ],
      mean_: [0.0, 0.0, 0.0, 0.0],
      std_: [1.0, 1.0, 1.0, 1.0],
    },
  },
  {
    key_: "nndeploy::track::FairMotPostProcess",
    name_: "FairMotPostProcess",
    desc_: "FairMot postprocess[device::Tensor->MOTResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Tensor",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    param_: {
      conf_thresh_: 0.4000000059604645,
      tracked_thresh_: 0.4000000059604645,
      min_box_area_: 200.0,
    },
  },
  {
    key_: "nndeploy::codec::BatchOpenCvEncode",
    name_: "BatchOpenCvEncode",
    desc_:
      "BatchOpenCvEncode node for encoding batches of images/videos using OpenCV",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "St6vectorIN2cv3MatESaIS1_EE",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [],
    node_key_: "",
  },
  {
    key_: "nndeploy::matting::PPMattingPostProcess",
    name_: "PPMattingPostProcess",
    desc_: "Matting postprocess[device::Tensor->MattingResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Tensor",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    param_: {
      alpha_h: 0,
      alpha_w: 0,
      output_h: 0,
      output_w: 0,
    },
  },
  {
    key_: "nndeploy::detect::DrawBox",
    name_: "DrawBox",
    desc_:
      "Draw detection boxes on input cv::Mat image based on detection results[cv::Mat->cv::Mat]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
      {
        type_: "Param",
        desc_: "input_1",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Mat",
        desc_: "output_0",
      },
    ],
  },
  {
    key_: "nndeploy::codec::OpenCvCameraDecode",
    name_: "OpenCvCameraDecode",
    desc_:
      "Decode camera stream using OpenCV, from camera device to cv::Mat frames, default color space is BGR",
    device_type_: "kDeviceTypeCodeCpu:0",
    is_dynamic_input_: false,
    inputs_: [],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Mat",
        desc_: "output_0",
      },
    ],
    flag_: "kCodecFlagImage",
    path_: "",
    size_: 0,
  },
  {
    key_: "CustomNode",
    name_: "CustomNode",
    desc_: "",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "torch.Tensor",
        desc_: "input_0",
      },
      {
        type_: "torch.Tensor",
        desc_: "input_1",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "torch.Tensor",
        desc_: "output_0",
      },
    ],
  },
  {
    key_: "nndeploy::detect::YoloXGraph",
    name_: "YoloXGraph",
    desc_:
      "cv::Mat to DetectResult[cv::Mat->preprocess->infer->postprocess->DetectResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    is_graph_: true,
    parallel_type_: "kParallelTypeNone",
    is_inner_: false,
    is_time_profile_: false,
    is_debug_: false,
    is_external_stream_: false,
    is_graph_node_share_stream_: true,
    queue_max_size_: 16,
    node_repository_: [
      {
        key_: "nndeploy::preprocess::CvtResizeNormTrans",
        name_: "preprocess",
        desc_:
          "cv::Mat to device::Tensor[cvtcolor->resize->normalize->transpose]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Mat",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        param_: {
          src_pixel_type_: "kPixelTypeGRAY",
          dst_pixel_type_: "kPixelTypeGRAY",
          interp_type_: "kInterpTypeNearst",
          h_: -1,
          w_: -1,
          data_type_: "kDataTypeCodeFp 32 1",
          data_format_: "kDataFormatNCHW",
          normalize_: true,
          scale_: [
            0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
            0.003921568859368563,
          ],
          mean_: [0.0, 0.0, 0.0, 0.0],
          std_: [1.0, 1.0, 1.0, 1.0],
        },
      },
      {
        key_: "nndeploy::infer::Infer",
        name_: "infer",
        desc_: "",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: true,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: true,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        type_: "kInferenceTypeNotSupport",
        param_: {
          inference_type_: "kInferenceTypeNone",
          model_type_: "kModelTypeDefault",
          is_path_: false,
          input_num_: 0,
          input_name_: [],
          input_shape_: [],
          output_num_: 0,
          output_name_: [],
          encrypt_type_: "kEncryptTypeNone",
          license_: "",
          device_type_: "kDeviceTypeCodeCpu:0",
          num_thread_: 1,
          gpu_tune_kernel_: 1,
          share_memory_mode_: "kShareMemoryTypeNoShare",
          precision_type_: "kPrecisionTypeFp32",
          power_type_: "kPowerTypeNormal",
          is_dynamic_shape_: false,
          min_shape_: {},
          opt_shape_: {},
          max_shape_: {},
          parallel_type_: "kParallelTypeNone",
          worker_num_: 4,
        },
      },
      {
        key_: "nndeploy::detect::YoloXPostProcess",
        name_: "postprocess",
        desc_: "YOLOX postprocess[device::Tensor->DetectResult]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Param",
            desc_: "output_0",
          },
        ],
        param_: {
          score_threshold_: 0.0,
          nms_threshold_: 0.0,
          num_classes_: 0,
          model_h_: 0,
          model_w_: 0,
        },
      },
    ],
  },
  {
    key_: "nndeploy::detect::YoloMultiConvOutputPostProcess",
    name_: "YoloMultiConvOutputPostProcess",
    desc_: "YOLO multi-conv output postprocess[device::Tensor->DetectResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Tensor",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    param_: {
      version_: -1,
      score_threshold_: 0.0,
      nms_threshold_: 0.0,
      obj_threshold_: 0.0,
      num_classes_: 0,
      model_h_: 0,
      model_w_: 0,
      anchors: [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
      ],
      strides: [8, 16, 32],
    },
  },
  {
    key_: "nndeploy::track::VisMOT",
    name_: "VisMOT",
    desc_:
      "Draw MOT result on input cv::Mat image based on MOT results[cv::Mat->cv::Mat]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
      {
        type_: "Param",
        desc_: "input_1",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Mat",
        desc_: "output_0",
      },
    ],
  },
  {
    key_: "nndeploy::detect::YoloGraph",
    name_: "YoloGraph",
    desc_:
      "yolo v5/v6/v7/v8/v11 graph[cv::Mat->preprocess->infer->postprocess->DetectResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    is_graph_: true,
    parallel_type_: "kParallelTypeNone",
    is_inner_: false,
    is_time_profile_: false,
    is_debug_: false,
    is_external_stream_: false,
    is_graph_node_share_stream_: true,
    queue_max_size_: 16,
    node_repository_: [
      {
        key_: "nndeploy::preprocess::CvtResizeNormTrans",
        name_: "preprocess",
        desc_:
          "cv::Mat to device::Tensor[cvtcolor->resize->normalize->transpose]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Mat",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        param_: {
          src_pixel_type_: "kPixelTypeGRAY",
          dst_pixel_type_: "kPixelTypeGRAY",
          interp_type_: "kInterpTypeNearst",
          h_: -1,
          w_: -1,
          data_type_: "kDataTypeCodeFp 32 1",
          data_format_: "kDataFormatNCHW",
          normalize_: true,
          scale_: [
            0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
            0.003921568859368563,
          ],
          mean_: [0.0, 0.0, 0.0, 0.0],
          std_: [1.0, 1.0, 1.0, 1.0],
        },
      },
      {
        key_: "nndeploy::infer::Infer",
        name_: "infer",
        desc_: "",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: true,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: true,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        type_: "kInferenceTypeNotSupport",
        param_: {
          inference_type_: "kInferenceTypeNone",
          model_type_: "kModelTypeDefault",
          is_path_: false,
          input_num_: 0,
          input_name_: [],
          input_shape_: [],
          output_num_: 0,
          output_name_: [],
          encrypt_type_: "kEncryptTypeNone",
          license_: "",
          device_type_: "kDeviceTypeCodeCpu:0",
          num_thread_: 1,
          gpu_tune_kernel_: 1,
          share_memory_mode_: "kShareMemoryTypeNoShare",
          precision_type_: "kPrecisionTypeFp32",
          power_type_: "kPowerTypeNormal",
          is_dynamic_shape_: false,
          min_shape_: {},
          opt_shape_: {},
          max_shape_: {},
          parallel_type_: "kParallelTypeNone",
          worker_num_: 4,
        },
      },
      {
        key_: "nndeploy::detect::YoloPostProcess",
        name_: "postprocess",
        desc_: "YOLO v5/v6/v7/v8/v11 postprocess[device::Tensor->DetectResult]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Param",
            desc_: "output_0",
          },
        ],
        param_: {
          version_: -1,
          score_threshold_: 0.0,
          nms_threshold_: 0.0,
          num_classes_: 0,
          model_h_: 0,
          model_w_: 0,
        },
      },
    ],
  },
  {
    key_: "nndeploy::codec::OpenCvImageEncode",
    name_: "OpenCvImageEncode",
    desc_:
      "Encode image using OpenCV, from cv::Mat to image file, supports common image formats",
    device_type_: "kDeviceTypeCodeCpu:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [],
    flag_: "kCodecFlagImage",
    path_: "",
    ref_path_: "",
    fourcc_: "mp4v",
    fps_: 0.0,
    width_: 0,
    height_: 0,
    size_: 0,
  },
  {
    key_: "nndeploy::codec::OpenCvImagesDecode",
    name_: "OpenCvImagesDecode",
    desc_:
      "Decode multiple images using OpenCV, from image paths to cv::Mat, default color space is BGR",
    device_type_: "kDeviceTypeCodeCpu:0",
    is_dynamic_input_: false,
    inputs_: [],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Mat",
        desc_: "output_0",
      },
    ],
    flag_: "kCodecFlagImage",
    path_: "",
    size_: 0,
  },
  {
    key_: "nndeploy::detect::YoloMultiConvDrawBox",
    name_: "YoloMultiConvDrawBox",
    desc_:
      "Draw detection boxes on input cv::Mat image based on detection results[cv::Mat->cv::Mat]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
      {
        type_: "Param",
        desc_: "input_1",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Mat",
        desc_: "output_0",
      },
    ],
  },
  {
    key_: "nndeploy::matting::VisMatting",
    name_: "VisMatting",
    desc_:
      "Draw matting result on input cv::Mat image based on matting results[cv::Mat->cv::Mat]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
      {
        type_: "Param",
        desc_: "input_1",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Mat",
        desc_: "output_0",
      },
    ],
  },
  {
    key_: "nndeploy::segment::SegmentRMBGGraph",
    name_: "SegmentRMBGGraph",
    desc_:
      "Segment RMBG graph[cv::Mat->preprocess->infer->postprocess->SegmentResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    is_graph_: true,
    parallel_type_: "kParallelTypeNone",
    is_inner_: false,
    is_time_profile_: false,
    is_debug_: false,
    is_external_stream_: false,
    is_graph_node_share_stream_: true,
    queue_max_size_: 16,
    node_repository_: [
      {
        key_: "nndeploy::preprocess::CvtResizeNormTrans",
        name_: "preprocess",
        desc_:
          "cv::Mat to device::Tensor[cvtcolor->resize->normalize->transpose]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Mat",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        param_: {
          src_pixel_type_: "kPixelTypeGRAY",
          dst_pixel_type_: "kPixelTypeGRAY",
          interp_type_: "kInterpTypeNearst",
          h_: -1,
          w_: -1,
          data_type_: "kDataTypeCodeFp 32 1",
          data_format_: "kDataFormatNCHW",
          normalize_: true,
          scale_: [
            0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
            0.003921568859368563,
          ],
          mean_: [0.0, 0.0, 0.0, 0.0],
          std_: [1.0, 1.0, 1.0, 1.0],
        },
      },
      {
        key_: "nndeploy::infer::Infer",
        name_: "infer",
        desc_: "",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: true,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: true,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        type_: "kInferenceTypeNone",
        param_: {
          inference_type_: "kInferenceTypeNone",
          model_type_: "kModelTypeDefault",
          is_path_: false,
          input_num_: 0,
          input_name_: [],
          input_shape_: [],
          output_num_: 0,
          output_name_: [],
          encrypt_type_: "kEncryptTypeNone",
          license_: "",
          device_type_: "kDeviceTypeCodeCpu:0",
          num_thread_: 1,
          gpu_tune_kernel_: 1,
          share_memory_mode_: "kShareMemoryTypeNoShare",
          precision_type_: "kPrecisionTypeFp32",
          power_type_: "kPowerTypeNormal",
          is_dynamic_shape_: false,
          min_shape_: {},
          opt_shape_: {},
          max_shape_: {},
          parallel_type_: "kParallelTypeNone",
          worker_num_: 4,
        },
      },
      {
        key_: "nndeploy::segment::RMBGPostProcess",
        name_: "postprocess",
        desc_: "Segment RMBG postprocess[device::Tensor->SegmentResult]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Param",
            desc_: "output_0",
          },
        ],
        param_: {
          version: -1,
        },
      },
    ],
  },
  {
    key_: "nndeploy::codec::OpenCvCameraEncode",
    name_: "OpenCvCameraEncode",
    desc_:
      "Encode camera stream using OpenCV, from cv::Mat frames to video output, supports common video formats",
    device_type_: "kDeviceTypeCodeCpu:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [],
    flag_: "kCodecFlagImage",
    path_: "",
    ref_path_: "",
    fourcc_: "mp4v",
    fps_: 0.0,
    width_: 0,
    height_: 0,
    size_: 0,
  },
  {
    key_: "nndeploy::classification::DrawLable",
    name_: "DrawLable",
    desc_:
      "Draw classification labels on input cv::Mat image based on classification results[cv::Mat->cv::Mat]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
      {
        type_: "Param",
        desc_: "input_1",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Mat",
        desc_: "output_0",
      },
    ],
  },
  {
    key_: "nndeploy::detect::YoloMultiConvOutputGraph",
    name_: "YoloMultiConvOutputGraph",
    desc_:
      "yolo multi-conv output graph[cv::Mat->preprocess->infer->postprocess->DetectResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    is_graph_: true,
    parallel_type_: "kParallelTypeNone",
    is_inner_: false,
    is_time_profile_: false,
    is_debug_: false,
    is_external_stream_: false,
    is_graph_node_share_stream_: true,
    queue_max_size_: 16,
    node_repository_: [
      {
        key_: "nndeploy::preprocess::WarpAffineCvtNormTrans",
        name_: "preprocess",
        desc_:
          "cv::Mat to device::Tensor[warpaffine->cvtcolor->normalize->transpose]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Mat",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        param_: {
          transform_: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          dst_w_: 0,
          dst_h_: 0,
          src_pixel_type_: "kPixelTypeGRAY",
          dst_pixel_type_: "kPixelTypeGRAY",
          data_type_: "kDataTypeCodeFp 32 1",
          data_format_: "kDataFormatNCHW",
          h_: -1,
          w_: -1,
          normalize_: true,
          const_value_: 114,
          scale_: [
            0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
            0.003921568859368563,
          ],
          mean_: [0.0, 0.0, 0.0, 0.0],
          std_: [1.0, 1.0, 1.0, 1.0],
          interp_type_: "kInterpTypeLinear",
          border_type_: "kBorderTypeConstant",
          border_val_: [0.0, 0.0, 0.0, 0.0],
        },
      },
      {
        key_: "nndeploy::infer::Infer",
        name_: "infer",
        desc_: "",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: true,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: true,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        type_: "kInferenceTypeNone",
        param_: {
          inference_type_: "kInferenceTypeNone",
          model_type_: "kModelTypeDefault",
          is_path_: false,
          input_num_: 0,
          input_name_: [],
          input_shape_: [],
          output_num_: 0,
          output_name_: [],
          encrypt_type_: "kEncryptTypeNone",
          license_: "",
          device_type_: "kDeviceTypeCodeCpu:0",
          num_thread_: 1,
          gpu_tune_kernel_: 1,
          share_memory_mode_: "kShareMemoryTypeNoShare",
          precision_type_: "kPrecisionTypeFp32",
          power_type_: "kPowerTypeNormal",
          is_dynamic_shape_: false,
          min_shape_: {},
          opt_shape_: {},
          max_shape_: {},
          parallel_type_: "kParallelTypeNone",
          worker_num_: 4,
        },
      },
      {
        key_: "nndeploy::detect::YoloMultiConvOutputPostProcess",
        name_: "postprocess",
        desc_:
          "YOLO multi-conv output postprocess[device::Tensor->DetectResult]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Param",
            desc_: "output_0",
          },
        ],
        param_: {
          version_: -1,
          score_threshold_: 0.0,
          nms_threshold_: 0.0,
          obj_threshold_: 0.0,
          num_classes_: 0,
          model_h_: 0,
          model_w_: 0,
          anchors: [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326],
          ],
          strides: [8, 16, 32],
        },
      },
    ],
  },
  {
    key_: "nndeploy::tokenizer::TokenizerDecodeCpp",
    name_: "TokenizerDecodeCpp",
    desc_:
      "A tokenizer decode node that uses the C++ tokenizers library to decode token IDs into text. Supports HuggingFace and BPE tokenizers. Can decode single token IDs or batches of token IDs. Provides token-to-text conversion.",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Param",
        desc_: "input_0",
      },
      {
        type_: "Param",
        desc_: "input_1",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
      {
        type_: "Param",
        desc_: "output_1",
      },
    ],
    param_: {
      is_path_: true,
      tokenizer_type_: "kTokenizerTypeHF",
      json_blob_: "",
      model_blob_: "",
      vocab_blob_: "",
      merges_blob_: "",
      added_tokens_: "",
      max_length_: 77,
    },
  },
  {
    key_: "nndeploy::codec::OpenCvImagesEncode",
    name_: "OpenCvImagesEncode",
    desc_:
      "Encode multiple images using OpenCV, from cv::Mat to image files, supports common image formats",
    device_type_: "kDeviceTypeCodeCpu:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [],
    flag_: "kCodecFlagImage",
    path_: "",
    ref_path_: "",
    fourcc_: "mp4v",
    fps_: 0.0,
    width_: 0,
    height_: 0,
    size_: 0,
  },
  {
    key_: "nndeploy::track::FairMotGraph",
    name_: "FairMotGraph",
    desc_: "FairMot graph[cv::Mat->preprocess->infer->postprocess->MOTResult]",
    device_type_: "kDeviceTypeCodeX86:0",
    is_dynamic_input_: false,
    inputs_: [
      {
        type_: "Mat",
        desc_: "input_0",
      },
    ],
    is_dynamic_output_: false,
    outputs_: [
      {
        type_: "Param",
        desc_: "output_0",
      },
    ],
    is_graph_: true,
    parallel_type_: "kParallelTypeNone",
    is_inner_: false,
    is_time_profile_: false,
    is_debug_: false,
    is_external_stream_: false,
    is_graph_node_share_stream_: true,
    queue_max_size_: 16,
    node_repository_: [
      {
        key_: "nndeploy::track::FairMotPreProcess",
        name_: "preprocess",
        desc_: "FairMot preprocess[cv::Mat->device::Tensor]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Mat",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
          {
            type_: "Tensor",
            desc_: "output_1",
          },
          {
            type_: "Tensor",
            desc_: "output_2",
          },
        ],
        param_: {
          src_pixel_type_: "kPixelTypeGRAY",
          dst_pixel_type_: "kPixelTypeGRAY",
          interp_type_: "kInterpTypeNearst",
          h_: -1,
          w_: -1,
          data_type_: "kDataTypeCodeFp 32 1",
          data_format_: "kDataFormatNCHW",
          normalize_: true,
          scale_: [
            0.003921568859368563, 0.003921568859368563, 0.003921568859368563,
            0.003921568859368563,
          ],
          mean_: [0.0, 0.0, 0.0, 0.0],
          std_: [1.0, 1.0, 1.0, 1.0],
        },
      },
      {
        key_: "nndeploy::infer::Infer",
        name_: "infer",
        desc_: "",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: true,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: true,
        outputs_: [
          {
            type_: "Tensor",
            desc_: "output_0",
          },
        ],
        type_: "kInferenceTypeNone",
        param_: {
          inference_type_: "kInferenceTypeNone",
          model_type_: "kModelTypeDefault",
          is_path_: false,
          input_num_: 0,
          input_name_: [],
          input_shape_: [],
          output_num_: 0,
          output_name_: [],
          encrypt_type_: "kEncryptTypeNone",
          license_: "",
          device_type_: "kDeviceTypeCodeCpu:0",
          num_thread_: 1,
          gpu_tune_kernel_: 1,
          share_memory_mode_: "kShareMemoryTypeNoShare",
          precision_type_: "kPrecisionTypeFp32",
          power_type_: "kPowerTypeNormal",
          is_dynamic_shape_: false,
          min_shape_: {},
          opt_shape_: {},
          max_shape_: {},
          parallel_type_: "kParallelTypeNone",
          worker_num_: 4,
        },
      },
      {
        key_: "nndeploy::track::FairMotPostProcess",
        name_: "post",
        desc_: "FairMot postprocess[device::Tensor->MOTResult]",
        device_type_: "kDeviceTypeCodeX86:0",
        is_dynamic_input_: false,
        inputs_: [
          {
            type_: "Tensor",
            desc_: "input_0",
          },
        ],
        is_dynamic_output_: false,
        outputs_: [
          {
            type_: "Param",
            desc_: "output_0",
          },
        ],
        param_: {
          conf_thresh_: 0.4000000059604645,
          tracked_thresh_: 0.4000000059604645,
          min_box_area_: 200.0,
        },
      },
    ],
  },
];
export const nodeHandler: MockItem[] = [
  // {
  //   url: "/node/branch",
  //   type: "post",
  //   response: (request) => {
  //     var params = JSON.parse(request.body);

  //     return {
  //       flag: "success",
  //       message: "成功",
  //       result: nodeBranches,
  //     };
  //   },
  // },
  // {
  //   url: "/node/tree",
  //   type: "post",
  //   response: (options) => {
  //     const data: INodeTreeNodeEntity[] = [
  //       ...nodeBranches.map((item) => ({ ...item, type: "branch" as const })),
  //       ...nodes.map((item) => ({ ...item, type: "leaf" as const })),
  //     ];

  //     return {
  //       flag: "success",
  //       message: "成功",
  //       result: data,
  //     };
  //   },
  // },
  // {
  //   url: "/node/branch/save",
  //   type: "post",
  //   response: (request: any) => {
  //     var entity = JSON.parse(request.body);

  //     const findIndex = nodeBranches.findIndex((item) => item.id == entity.id);

  //     if (findIndex == -1) {
  //       entity.id = Random.guid();
  //       nodeBranches.push(entity);
  //     } else {
  //       nodeBranches[findIndex] = entity;
  //     }

  //     return {
  //       flag: "success",
  //       message: "成功",
  //       result: entity,
  //     };
  //   },
  // },
  // {
  //   url: "/node/branch/delete",
  //   type: "post",
  //   response: (options) => {
  //     const entity: INodeBranchEntity = JSON.parse(options.body);

  //     const findIndex = nodeBranches.findIndex((item) => item.id == entity.id);

  //     if (findIndex == -1) {
  //       return {
  //         flag: "error",
  //         message: "could not find this item",
  //         result: {},
  //       };
  //     } else {
  //       nodeBranches.splice(findIndex, 1);
  //     }

  //     return {
  //       flag: "success",
  //       message: "",
  //       result: {},
  //     };
  //   },
  // },

  // {
  //   url: "/node/save",
  //   type: "post",
  //   response: (request: any) => {
  //     var entity = JSON.parse(request.body);

  //     const findIndex = nodes.findIndex((item) => item.key_ == entity.id);

  //     if (findIndex == -1) {
  //       entity.id = Random.guid();
  //       nodes.push(entity);
  //     } else {
  //       nodes[findIndex] = entity;
  //     }

  //     return {
  //       flag: "success",
  //       message: "成功",
  //       result: entity,
  //     };
  //   },
  // },
  // {
  //   url: "/node/delete",
  //   type: "post",
  //   response: (options) => {
  //     const entity: INodeEntity = JSON.parse(options.body);

  //     const findIndex = nodes.findIndex((item) => item.key_ == entity.key_);

  //     if (findIndex == -1) {
  //       return {
  //         flag: "error",
  //         message: "could not find this item",
  //         result: {},
  //       };
  //     } else {
  //       nodes.splice(findIndex, 1);
  //     }

  //     return {
  //       flag: "success",
  //       message: "",
  //       result: {},
  //     };
  //   },
  // },
  {
    url: "/node/list",
    type: "post",
    response: (options) => {
      return {
        flag: "success",
        message: "",
        result: nodes,
      };
    },
  },

  {
    url: "/node/get",
    type: "post",
    response: (options) => {
      const requestParams: INodeEntity = JSON.parse(options.body);
      let entity = nodes.find((item) => item.key_ == requestParams.key_);
      return {
        flag: "success",
        message: "",
        result: entity,
      };
    },
  },

  // {
  //   url: "/node/page",
  //   type: "post",
  //   response: (options) => {
  //     const {
  //       parentId,
  //       currentPage,
  //       pageSize,
  //     }: { parentId: string; currentPage: number; pageSize: number } =
  //       JSON.parse(options.body);

  //     let finds: INodeEntity[] = [];
  //     if (parentId) {
  //       finds = nodes.filter((item) => item.inputs_ == parentId);
  //     } else {
  //       finds = nodes;
  //     }

  //     const offset = (currentPage - 1) * pageSize;

  //     const records = finds.slice(offset, pageSize);

  //     return {
  //       flag: "success",
  //       message: "",
  //       result: {
  //         records,
  //         total: finds.length,
  //       },
  //     };
  //   },
  // },
];
