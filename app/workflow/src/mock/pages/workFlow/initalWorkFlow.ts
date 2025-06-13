import { IWorkFlowEntity } from "../../../pages/Layout/Design/WorkFlow/entity";

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
