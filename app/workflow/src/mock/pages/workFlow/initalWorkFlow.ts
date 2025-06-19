import {
  IBusinessNode,
  IWorkFlowEntity,
} from "../../../pages/Layout/Design/WorkFlow/entity";

export const workFlows: IWorkFlowEntity[] = [
  {
    id: "1",
    name: "demo",
    parentId: "",
    businessContent: {
      key_: "nndeploy::dag::Graph",
      name_: "demo",
      device_type_: "kDeviceTypeCodeX86:0",
      inputs_: [],
      outputs_: [
        {
          name_: "detect_out",
          type_: "kNotSet",
        },
      ],
      is_external_stream_: false,
      is_inner_: false,
      is_time_profile_: true,
      is_debug_: false,
      is_graph_node_share_stream_: true,
      queue_max_size_: 16,
      node_repository_: [
        {
          key_: "nndeploy::detect::YoloGraph",
          name_: "nndeploy::detect::YoloGraph",
          device_type_: "kDeviceTypeCodeX86:0",
          inputs_: [
            {
              name_: "detect_in",
              type_: "CvMat",
            },
          ],
          outputs_: [
            {
              name_: "detect_out",
              type_: "Param",
            },
          ],
          parallel_type_: "kParallelTypeSequential",
          node_repository_: [
            {
              key_: "nndeploy::preprocess::CvtColorResize",
              name_: "preprocess",
              device_type_: "kDeviceTypeCodeX86:0",
              inputs_: [
                {
                  name_: "detect_in",
                  type_: "CvMat",
                },
              ],
              outputs_: [
                {
                  name_: "images",
                  type_: "Tensor",
                },
              ],
              param_: {
                src_pixel_type_: "kPixelTypeBGR",
                dst_pixel_type_: "kPixelTypeRGB",
                interp_type_: "kInterpTypeLinear",
                h_: 640,
                w_: 640,
                data_type_: "kDataTypeCodeFp 32 1",
                data_format_: "kDataFormatNCHW",
                normalize_: true,
                scale_: [
                  0.003921568859368563, 0.003921568859368563,
                  0.003921568859368563, 0.003921568859368563,
                ],
                mean_: [0.0, 0.0, 0.0, 0.0],
                std_: [1.0, 1.0, 1.0, 1.0],
              },
            },
            {
              key_: "nndeploy::infer::Infer",
              name_: "infer",
              device_type_: "kDeviceTypeCodeX86:0",
              inputs_: [
                {
                  name_: "images",
                  type_: "Tensor",
                },
              ],
              outputs_: [
                {
                  name_: "output0",
                  type_: "Tensor",
                },
              ],
              type_: "kInferenceTypeOnnxRuntime",
              is_input_dynamic_: false,
              is_output_dynamic_: false,
              can_op_input_: false,
              can_op_output_: false,
              param_: {
                inference_type_: 6,
                model_type_: 5,
                is_path_: true,
                model_value_: ["yolo11s.sim.onnx"],
                input_num_: 0,
                input_name_: [],
                input_shape_: [],
                output_num_: 0,
                output_name_: [],
                encrypt_type_: 0,
                license_: "",
                device_type_: "kDeviceTypeCodeX86:0",
                num_thread_: 4,
                gpu_tune_kernel_: 1,
                share_memory_mode_: 0,
                precision_type_: 2,
                power_type_: 1,
                is_dynamic_shape_: false,
                parallel_type_: 2,
                worker_num_: 4,
              },
            },
            {
              key_: "nndeploy::detect::YoloPostProcess",
              name_: "postprocess",
              device_type_: "kDeviceTypeCodeX86:0",
              inputs_: [
                {
                  name_: "output0",
                  type_: "Tensor",
                },
              ],
              outputs_: [
                {
                  name_: "detect_out",
                  type_: "Param",
                },
              ],
              param_: {},
            },
          ],
        },
        {
          key_: "nndeploy::codec::OpenCvImageDecodeNode",
          name_: "decode_node",
          device_type_: "kDeviceTypeCodeCpu:0",
          inputs_: [],
          outputs_: [
            {
              name_: "detect_in",
              type_: "CvMat",
            },
          ],
          flag_: "kCodecFlagImage",
        },
        {
          key_: "nndeploy::detect::DrawBoxNode",
          name_: "DrawBoxNode",
          device_type_: "kDeviceTypeCodeX86:0",
          inputs_: [
            {
              name_: "detect_in",
              type_: "CvMat",
            },
            {
              name_: "detect_out",
              type_: "Param",
            },
          ],
          outputs_: [
            {
              name_: "draw_output",
              type_: "CvMat",
            },
          ],
        },
        {
          key_: "nndeploy::codec::OpenCvImageEncodeNode",
          name_: "encode_node",
          device_type_: "kDeviceTypeCodeCpu:0",
          inputs_: [
            {
              name_: "draw_output",
              type_: "CvMat",
            },
          ],
          outputs_: [],
          flag_: "kCodecFlagImage",
        },
      ],
    },
    designContent: {
      nodes: [],
      edges: [],
    },
  },
];

export const businessContents: IBusinessNode[] = [
  {
    key_: "nndeploy::dag::Graph",
    name_: "flow1",
    device_type_: "kDeviceTypeCodeX86:0",
    inputs_: [],
    outputs_: [
      {
        name_: "detect_out",
        type_: "kNotSet",
      },
    ],
    is_external_stream_: false,
    is_inner_: false,
    "is_time_profile_": true,
    is_debug_: false,
    is_graph_node_share_stream_: true,
    queue_max_size_: 16,
    node_repository_: [
      {
        key_: "nndeploy::codec::OpenCvImageDecodeNode",
        name_: "decode_node_0",
        device_type_: "kDeviceTypeCodeCpu:0",
        inputs_: [],
        outputs_: [
          {
            desc_: "detect_in",
            type_: "CvMat",
            id: "portydxz0apy4",
            name_: "decode_node_0@preprocess_2@detect_in@detect_in",
          },
          {
            desc_: "detect_in",
            type_: "CvMat",
            id: "portydxz0apy4",
            name_: "decode_node_0@DrawBoxNode_5@detect_in@detect_in",
          },
        ],
        flag_: "kCodecFlagImage",
        param_: {},
        node_repository_: [],
      },
      {
        key_: "nndeploy::detect::YoloGraph",
        name_: "nndeploy::detect::YoloGraph_1",
        device_type_: "kDeviceTypeCodeX86:0:0",
        is_graph_: true, 
        inputs_: [
          {
            desc_: "detect_in",
            type_: "CvMat",
            id: "port9pzze8b3y",
          },
        ],
        outputs_: [
          {
            desc_: "detect_out",
            type_: "Param",
            id: "portbin37q37u",
          },
        ],
        param_: {},
        node_repository_: [
          {
            key_: "nndeploy::preprocess::CvtColorResize",
            name_: "preprocess_2",
            device_type_: "kDeviceTypeCodeX86:0",
            inputs_: [
              {
                desc_: "detect_in",
                type_: "Mat",
                id: "port3t7w8qgew",
                name_: "decode_node_0@preprocess_2@detect_in@detect_in",
              },
            ],
            outputs_: [
              {
                desc_: "images",
                type_: "Tensor",
                id: "port5lllj3tel",
                name_: "preprocess_2@infer_3@images@input0",
              },
            ],
            param_: {
              src_pixel_type_: "kPixelTypeBGR",
              dst_pixel_type_: "kPixelTypeRGB",
              interp_type_: "kInterpTypeLinear",
              h_: 640,
              w_: 640,
              data_type_: "kDataTypeCodeFp 32 1",
              data_format_: "kDataFormatNCHW",
              normalize_: true,
              scale_: [
                0.003921568859368563, 0.003921568859368563,
                0.003921568859368563, 0.003921568859368563,
              ],
              mean_: [0, 0, 0, 0],
              std_: [1, 1, 1, 1],
            },
            node_repository_: [],
          },
          {
            key_: "nndeploy::infer::Infer",
            name_: "infer_3",
            device_type_: "kDeviceTypeCodeX86:0",
            inputs_: [
              {
                desc_: "input0",
                type_: "Tensor",
                id: "portbbug0fy5d",
                name_: "preprocess_2@infer_3@images@input0",
              },
            ],
            outputs_: [
              {
                desc_: "output0",
                type_: "Tensor",
                id: "port3mrkkq9ww",
                name_: "infer_3@postprocess_4@output0@output0",
              },
            ],
            is_dynamic_input_: true,
            is_dynamic_output_: true,
            type_: "kInferenceTypeOnnxRuntime",
            param_: {
              inference_type_: "kInferenceTypeOnnxRuntime",
              model_type_: "kModelTypeOnnx",
              is_path_: true,
              model_value_: ["yolo11s.sim.onnx"],
              input_num_: 0,
              input_name_: [],
              input_shape_: [],
              output_num_: 0,
              output_name_: [],
              encrypt_type_: "kEncryptTypeNone",
              license_: "",
              device_type_: "kDeviceTypeCodeX86:0",
              num_thread_: 4,
              gpu_tune_kernel_: 1,
              share_memory_mode_: "kShareMemoryTypeNoShare",
              precision_type_: "kPrecisionTypeFp32",
              power_type_: "kPowerTypeNormal",
              is_dynamic_shape_: false,
              min_shape_: {},
              opt_shape_: {},
              max_shape_: {},
              parallel_type_: "kParallelTypeSequential",
              worker_num_: 4,
            },
            node_repository_: [],
          },
          {
            key_: "nndeploy::detect::YoloPostProcess",
            name_: "postprocess_4",
            device_type_: "kDeviceTypeCodeX86:0",
            inputs_: [
              {
                desc_: "output0",
                type_: "Tensor",
                id: "port0qz55ty3n",
                name_: "infer_3@postprocess_4@output0@output0",
              },
            ],
            outputs_: [
              {
                desc_: "detect_out",
                type_: "Param",
                id: "porthyeq6xt7z",
                name_: "postprocess_4@DrawBoxNode_5@detect_out@detect_out",
              },
            ],
            param_: {
              version_: 11,
              score_threshold_: 0.5,
              nms_threshold_: 0.44999998807907104,
              num_classes_: 80,
              model_h_: 640,
              model_w_: 640,
            },
            node_repository_: [],
          },
        ],
      },
      {
        key_: "nndeploy::detect::DrawBoxNode",
        name_: "DrawBoxNode_5",
        device_type_: "kDeviceTypeCodeX86:0",
        inputs_: [
          {
            desc_: "detect_in",
            type_: "CvMat",
            id: "portue58lowg4",
            name_: "decode_node_0@DrawBoxNode_5@detect_in@detect_in",
          },
          {
            desc_: "detect_out",
            type_: "Param",
            id: "portcsge0idz1",
            name_: "postprocess_4@DrawBoxNode_5@detect_out@detect_out",
          },
        ],
        outputs_: [
          {
            desc_: "draw_output",
            type_: "CvMat",
            id: "portaw76g8uqe",
            name_: "DrawBoxNode_5@encode_node_6@draw_output@draw_output",
          },
        ],
        param_: {},
        node_repository_: [],
      },
      {
        key_: "nndeploy::codec::OpenCvImageEncodeNode",
        name_: "encode_node_6",
        device_type_: "kDeviceTypeCodeCpu:0",
        inputs_: [
          {
            desc_: "draw_output",
            type_: "CvMat",
            id: "porth10422wn2",
            name_: "DrawBoxNode_5@encode_node_6@draw_output@draw_output",
          },
        ],
        outputs_: [],
        flag_: "kCodecFlagImage",
        param_: {},
        node_repository_: [],
      },
    ],
  },
];
