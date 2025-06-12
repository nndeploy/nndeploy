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

const nodes: INodeEntity[] = [
        {
            "key_": "nndeploy::preprocess::CvtColorResize",
            "name_": "preprocess",
            "device_type_": "kDeviceTypeCodeX86:0",
            "inputs_": [
                {
                    "name_": "detect_in",
                    "type_": "Mat"
                }, 
                 {
                    "name_": "detect_in2",
                    "type_": "Mat"
                }
            ],
            "outputs_": [
                {
                    "name_": "images",
                    "type_": "Tensor"
                }
            ],
            "param_": {
                "src_pixel_type_": "kPixelTypeBGR",
                "dst_pixel_type_": "kPixelTypeRGB",
                "interp_type_": "kInterpTypeLinear",
                "h_": 640,
                "w_": 640,
                "data_type_": "kDataTypeCodeFp 32 1",
                "data_format_": "kDataFormatNCHW",
                "normalize_": true,
                "scale_": [
                    0.003921568859368563,
                    0.003921568859368563,
                    0.003921568859368563,
                    0.003921568859368563
                ],
                "mean_": [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                "std_": [
                    1.0,
                    1.0,
                    1.0,
                    1.0
                ]
            }
        },
        {
            "key_": "nndeploy::infer::Infer",
            "name_": "infer",
            "device_type_": "kDeviceTypeCodeX86:0",
            "inputs_": [
                {
                    "name_": "input0",
                    "type_": "Tensor"
                }
            ],
            "outputs_": [
                {
                    "name_": "output0",
                    "type_": "Tensor"
                }
            ],
            "type_": "kInferenceTypeOnnxRuntime",
            "param_": {
                "inference_type_": "kInferenceTypeOnnxRuntime",
                "model_type_": "kModelTypeOnnx",
                "is_path_": true,
                "model_value_": [
                    "yolo11s.sim.onnx"
                ],
                "input_num_": 0,
                "input_name_": [],
                "input_shape_": [],
                "output_num_": 0,
                "output_name_": [],
                "encrypt_type_": "kEncryptTypeNone",
                "license_": "",
                "device_type_": "kDeviceTypeCodeX86:0",
                "num_thread_": 4,
                "gpu_tune_kernel_": 1,
                "share_memory_mode_": "kShareMemoryTypeNoShare",
                "precision_type_": "kPrecisionTypeFp32",
                "power_type_": "kPowerTypeNormal",
                "is_dynamic_shape_": false,
                "min_shape_": {},
                "opt_shape_": {},
                "max_shape_": {},
                "parallel_type_": "kParallelTypeSequential",
                "worker_num_": 4
            }
        },
        {
            "key_": "nndeploy::detect::YoloPostProcess",
            "name_": "postprocess",
            "device_type_": "kDeviceTypeCodeX86:0",
            "inputs_": [
                {
                    "name_": "output0",
                    "type_": "Tensor"
                }
            ],
            "outputs_": [
                {
                    "name_": "detect_out",
                    "type_": "Param"
                }
            ],
            "param_": {
                "version_": 11,
                "score_threshold_": 0.5,
                "nms_threshold_": 0.44999998807907104,
                "num_classes_": 80,
                "model_h_": 640,
                "model_w_": 640
            }
        }
    ]

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
