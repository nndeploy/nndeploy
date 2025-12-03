# flowai.gram流程文档等与后端流程图文档之间的转换
##	转换原因 
字节的流程引擎flowgram 定义了流程的的文档数据结构, 但后端业务使用的是自己的文档数据结构, 需要在两者之间做转化.  在保存flowgram流程图时将flowgram的文档转化为后端文档结构, 加载后端的流程时将后端的文档结构转换为flowgram的文档结构 
##	结构差异

 我们以Detect_Yoyo为例, 可观察到两个模型文档之间的如下区别

- **节点区别**

  - flowgram的文档通过block字段嵌套子节点, 大小位置信息保存在节点中的meta字段中, 每个节点都有唯一的键值id, 保存为后端文档时, 这个字段用后端的节点的name_字段替换作为节点的唯一标识. 

  - 后端文档通过node_repository_嵌套子节点, 所有节点的位置大小信息保存到全局的nndeploy_ui_layout中. 每个节点的键是name_字段, 前端加载文档时会生成随机的id值作为节点的标志. 

- **线条区别**
  - flowgram的线条结构体为WorkflowEdgeJSON, 源节点, 源端口, 目标节点, 目标端口这四个字段确定一条线条, 后端的文档没有线条的概念, 后端文档通过节点的name_字段连接线条, 节点端口用放到input与output字段中, 当output字段里面的name_值input字段字段的name_值一致时表明两个节点的对应的端口之间有连线, 前端加载保存的后端流程图是, 根据name_绘制出出相应的连线. 

- **流程图**

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/image/workflow.png">
    <img alt="nndeploy" src="https://github.com/nndeploy/nndeploy/tree/main/app/workflow/src/assets/readme/detect-yolo.png" width=100%>
  </picture>
</p>

- **flowgram文档数据结构**

  - 工作流线条

```
interface WorkflowEdgeJSON {
    sourceNodeID: string; //源节点
    targetNodeID: string; //目标节点
    sourcePortID?: string | number; //源端口
    targetPortID?: string | number;//目标端口
    data?: any;
}
```

  - 工作流点

```interface WorkflowNodeJSON extends FlowNodeJSON {
    id: string;
    type: string | number;
    /**
     * ui 数据
     */
    meta?: WorkflowNodeMeta;
    /**
     * 表单数据, 业务自定义
     */
    data?: any;
    /**
     * 子节点
     */
    blocks?: WorkflowNodeJSON[];
    /**
     * 子节点间连线
     */
    edges?: WorkflowEdgeJSON[];
}
```

- 工作流文档结构体

```
export interface FlowDocumentJSON {
  nodes: FlowNodeJSON[];  //节点数组
  edges: WorkflowEdgeJSON[]; //边数组
}
```
 - 上图 Detect_Yoyo 流程的flowgram文档结构
 ```
{
    "nodes": [
        {
            "id": "node_xa9or5895",
            "type": "nndeploy::preprocess::CvtResizeNormTrans",
            "meta": {
                "position": {
                    "x": 380,
                    "y": 104.4
                }
            },
            "data": {
                "key_": "nndeploy::preprocess::CvtResizeNormTrans",
                "name_": "CvtResizeNormTrans_17",
                "desc_": "cv::Mat to device::Tensor[cvtcolor->resize->normalize->transpose]",
                "device_type_": "kDeviceTypeCodeCpu:0",
                "is_dynamic_input_": false,
                "inputs_": [
                    {
                        "id": "port_46p1dlfcf",
                        "desc_": "input_0",
                        "name_": "OpenCvImageDecode_22@output_0",
                        "type_": "ndarray"
                    }
                ],
                "is_dynamic_output_": false,
                "outputs_": [
                    {
                        "id": "port_9t4e3y2wz",
                        "desc_": "output_0",
                        "name_": "CvtResizeNormTrans_17@output_0",
                        "type_": "Tensor"
                    }
                ],
                "node_type_": "Intermediate",
                "param_": {
                    "src_pixel_type_": "kPixelTypeBGR",
                    "dst_pixel_type_": "kPixelTypeRGB",
                    "interp_type_": "kInterpTypeLinear",
                    "h_": 640,
                    "w_": 640,
                    "data_type_": "kDataTypeCodeFp32",
                    "data_format_": "kDataFormatNCHW",
                    "normalize_": true,
                    "scale_": [
                        0.003921568859368563,
                        0.003921568859368563,
                        0.003921568859368563,
                        0.003921568859368563
                    ],
                    "mean_": [
                        0,
                        0,
                        0,
                        0
                    ],
                    "std_": [
                        1,
                        1,
                        1,
                        1
                    ]
                },
                "size": {
                    "width": 200,
                    "height": 80
                },
                "id": "node_xa9or5895"
            }
        },
        {
            "id": "node_y2rg4k1di",
            "type": "nndeploy::infer::Infer",
            "meta": {
                "position": {
                    "x": 690,
                    "y": 104.40000000000003
                }
            },
            "data": {
                "key_": "nndeploy::infer::Infer",
                "name_": "Infer_18",
                "desc_": "Universal Inference Node - Enables cross-platform model deployment with multiple inference backends while maintaining native performance",
                "device_type_": "kDeviceTypeCodeCpu:0",
                "is_dynamic_input_": true,
                "inputs_": [
                    {
                        "id": "port_hqdfvw674",
                        "desc_": "input_0",
                        "name_": "CvtResizeNormTrans_17@output_0",
                        "type_": "Tensor"
                    }
                ],
                "is_dynamic_output_": true,
                "outputs_": [
                    {
                        "id": "port_t666x2pvc",
                        "desc_": "output_0",
                        "name_": "Infer_18@output_0",
                        "type_": "Tensor"
                    }
                ],
                "node_type_": "Intermediate",
                "type_": "kInferenceTypeOnnxRuntime",
                "param_": {
                    "model_type_": "kModelTypeOnnx",
                    "is_path_": true,
                    "model_value_": [
                        "resources/models/detect/yolo11s.sim.onnx"
                    ],
                    "external_model_data_": [
                        ""
                    ],
                    "device_type_": "kDeviceTypeCodeCpu:0",
                    "num_thread_": 8,
                    "gpu_tune_kernel_": 1,
                    "input_num_": 1,
                    "input_name_": [
                        ""
                    ],
                    "input_shape_": [
                        [
                            -1,
                            -1,
                            -1,
                            -1
                        ]
                    ],
                    "output_num_": 1,
                    "output_name_": [
                        ""
                    ],
                    "encrypt_type_": "kEncryptTypeNone",
                    "license_": "",
                    "share_memory_mode_": "kShareMemoryTypeNoShare",
                    "precision_type_": "kPrecisionTypeFp32",
                    "power_type_": "kPowerTypeNormal",
                    "is_dynamic_shape_": false,
                    "min_shape_": {
                        "input_0": [
                            -1,
                            -1,
                            -1,
                            -1
                        ]
                    },
                    "opt_shape_": {
                        "input_0": [
                            -1,
                            -1,
                            -1,
                            -1
                        ]
                    },
                    "max_shape_": {
                        "input_0": [
                            -1,
                            -1,
                            -1,
                            -1
                        ]
                    },
                    "parallel_type_": "kParallelTypeNone",
                    "worker_num_": 1
                },
                "size": {
                    "width": 200,
                    "height": 80
                },
                "id": "node_y2rg4k1di"
            }
        },
        {
            "id": "node_t20tp8z8d",
            "type": "nndeploy::detect::YoloPostProcess",
            "meta": {
                "position": {
                    "x": 982,
                    "y": 104.40000000000002
                }
            },
            "data": {
                "key_": "nndeploy::detect::YoloPostProcess",
                "name_": "YoloPostProcess_19",
                "desc_": "YOLO v5/v6/v7/v8/v11 postprocess[device::Tensor->DetectResult]",
                "device_type_": "kDeviceTypeCodeCpu:0",
                "is_dynamic_input_": false,
                "inputs_": [
                    {
                        "id": "port_i7a3cbqns",
                        "desc_": "input_0",
                        "name_": "Infer_18@output_0",
                        "type_": "Tensor"
                    }
                ],
                "is_dynamic_output_": false,
                "outputs_": [
                    {
                        "id": "port_dmx7mrdcv",
                        "desc_": "output_0",
                        "name_": "YoloPostProcess_19@output_0",
                        "type_": "DetectResult"
                    }
                ],
                "node_type_": "Intermediate",
                "param_": {
                    "version_": 11,
                    "score_threshold_": 0.5,
                    "nms_threshold_": 0.45,
                    "num_classes_": 80,
                    "model_h_": 640,
                    "model_w_": 640
                },
                "size": {
                    "width": 200,
                    "height": 80
                },
                "id": "node_t20tp8z8d"
            }
        },
        {
            "id": "node_fig27irfm",
            "type": "nndeploy::detect::DrawBox",
            "meta": {
                "position": {
                    "x": 1274,
                    "y": 7.700000000000017
                }
            },
            "data": {
                "key_": "nndeploy::detect::DrawBox",
                "name_": "DrawBox_20",
                "desc_": "Draw detection boxes on input cv::Mat image based on detection results[cv::Mat->cv::Mat]",
                "device_type_": "kDeviceTypeCodeCpu:0",
                "is_dynamic_input_": false,
                "inputs_": [
                    {
                        "id": "port_gjq837uhv",
                        "desc_": "input_0",
                        "name_": "OpenCvImageDecode_22@output_0",
                        "type_": "ndarray"
                    },
                    {
                        "id": "port_9ubisk0g9",
                        "desc_": "input_1",
                        "name_": "YoloPostProcess_19@output_0",
                        "type_": "DetectResult"
                    }
                ],
                "is_dynamic_output_": false,
                "outputs_": [
                    {
                        "id": "port_tlgb6gv59",
                        "desc_": "output_0",
                        "name_": "DrawBox_20@output_0",
                        "type_": "ndarray"
                    }
                ],
                "node_type_": "Intermediate",
                "size": {
                    "width": 200,
                    "height": 80
                },
                "id": "node_fig27irfm"
            }
        },
        {
            "id": "node_ujy9hq0ta",
            "type": "nndeploy::codec::OpenCvImageDecode",
            "meta": {
                "position": {
                    "x": 100,
                    "y": 0
                }
            },
            "data": {
                "key_": "nndeploy::codec::OpenCvImageDecode",
                "name_": "OpenCvImageDecode_22",
                "developer_": "",
                "source_": "",
                "desc_": "Decode image using OpenCV, from image path to cv::Mat, default color space is BGR",
                "device_type_": "kDeviceTypeCodeCpu:0",
                "version_": "1.0.0",
                "required_params_": [
                    "path_"
                ],
                "ui_params_": [],
                "is_dynamic_input_": false,
                "inputs_": [],
                "is_dynamic_output_": false,
                "outputs_": [
                    {
                        "id": "port_em791qwj7",
                        "desc_": "output_0",
                        "name_": "OpenCvImageDecode_22@output_0",
                        "type_": "ndarray"
                    }
                ],
                "node_type_": "Input",
                "io_type_": "Image",
                "path_": "resources/template/nndeploy-workflow/detect/zidane.jpg",
                "size": {
                    "width": 200,
                    "height": 80
                },
                "id": "node_ujy9hq0ta"
            }
        },
        {
            "id": "node_v0kyvlwfh",
            "type": "nndeploy::codec::OpenCvImageEncode",
            "meta": {
                "position": {
                    "x": 1569,
                    "y": 7.700000000000017
                }
            },
            "data": {
                "key_": "nndeploy::codec::OpenCvImageEncode",
                "name_": "OpenCvImageEncode_23",
                "developer_": "",
                "source_": "",
                "desc_": "Encode image using OpenCV, from cv::Mat to image file, supports common image formats",
                "device_type_": "kDeviceTypeCodeCpu:0",
                "version_": "1.0.0",
                "required_params_": [
                    "path_"
                ],
                "ui_params_": [],
                "is_dynamic_input_": false,
                "inputs_": [
                    {
                        "id": "port_oy9sbivvk",
                        "desc_": "input_0",
                        "name_": "DrawBox_20@output_0",
                        "type_": "ndarray"
                    }
                ],
                "is_dynamic_output_": false,
                "outputs_": [],
                "node_type_": "Output",
                "io_type_": "Image",
                "path_": "resources/images/result.yolo.jpg",
                "size": {
                    "width": 200,
                    "height": 80
                },
                "id": "node_v0kyvlwfh"
            }
        }
    ],
    "edges": [
        {
            "sourceNodeID": "node_ujy9hq0ta",
            "targetNodeID": "node_xa9or5895",
            "sourcePortID": "port_em791qwj7",
            "targetPortID": "port_46p1dlfcf"
        },
        {
            "sourceNodeID": "node_xa9or5895",
            "targetNodeID": "node_y2rg4k1di",
            "sourcePortID": "port_9t4e3y2wz",
            "targetPortID": "port_hqdfvw674"
        },
        {
            "sourceNodeID": "node_y2rg4k1di",
            "targetNodeID": "node_t20tp8z8d",
            "sourcePortID": "port_t666x2pvc",
            "targetPortID": "port_i7a3cbqns"
        },
        {
            "sourceNodeID": "node_t20tp8z8d",
            "targetNodeID": "node_fig27irfm",
            "sourcePortID": "port_dmx7mrdcv",
            "targetPortID": "port_9ubisk0g9"
        },
        {
            "sourceNodeID": "node_ujy9hq0ta",
            "targetNodeID": "node_fig27irfm",
            "sourcePortID": "port_em791qwj7",
            "targetPortID": "port_gjq837uhv"
        },
        {
            "sourceNodeID": "node_fig27irfm",
            "targetNodeID": "node_v0kyvlwfh",
            "sourcePortID": "port_tlgb6gv59",
            "targetPortID": "port_oy9sbivvk"
        }
    ]
}
```

- **后端流程数据结构**

  - 节点位置大小结构体

```
export interface Inndeploy_ui_layout {
  // position?: { x: number, y: number },
  // size?: { width: number, height: number },
  layout: {
    [nodeName: string]: {
      //config: INodeUiExtraInfo, 
      expanded?: boolean,
      position?: { x: number, y: number },
      size?: { width: number, height: number },
      children?: { [nodeName: string]: INodeUiExtraInfo }


    }
  },
  //nodeExtra: { [nodeName: string]: INodeUiExtraInfo }
  groups: { name_: string, blockIDs: string[], parentID: string,  }[]
}
```

- 节点结构体, 通过node_repository_嵌套子节点

```
export interface IBusinessNode {
  key_: string;
  name_: string;
  desc_: string;
  device_type_: string;
  inputs_: IConnectinPoint[],
  outputs_: IConnectinPoint[],
  node_repository_?: IBusinessNode[],
  [key: string]: any;
  nndeploy_ui_layout?: Inndeploy_ui_layout
}
```
- 上图 Detect_Yoyo 流程的文档结构
```
{
    "key_": "nndeploy.dag.Graph",
    "name_": "Detect_YOLO",
    "developer_": "Always",
    "source_": "https://github.com/ultralytics/ultralytics",
    "desc_": "YOLO-based object detection workflow for identifying and locating multiple objects in images",
    "device_type_": "kDeviceTypeCodeCpu:0",
    "is_dynamic_input_": false,
    "inputs_": [],
    "is_dynamic_output_": false,
    "outputs_": [],
    "is_graph_": true,
    "parallel_type_": "kParallelTypeNone",
    "is_inner_": false,
    "node_type_": "Intermediate",
    "is_time_profile_": false,
    "is_debug_": false,
    "is_external_stream_": false,
    "is_graph_node_share_stream_": true,
    "queue_max_size_": 16,
    "is_loop_max_flag_": true,
    "loop_count_": -1,
    "image_url_": [
        "template[http,modelscope]@https://template.cn/template.jpg"
    ],
    "video_url_": [
        "template[http,modelscope]@https://template.cn/template.mp4"
    ],
    "audio_url_": [
        "template[http,modelscope]@https://template.cn/template.mp3"
    ],
    "model_url_": [
        "modelscope@nndeploy/nndeploy:detect/yolo11s.sim.onnx"
    ],
    "other_url_": [
        "template[http,modelscope]@https://template.cn/template.txt"
    ],
    "node_repository_": [
        {
            "key_": "nndeploy::preprocess::CvtResizeNormTrans",
            "name_": "CvtResizeNormTrans_17",
            "desc_": "cv::Mat to device::Tensor[cvtcolor->resize->normalize->transpose]",
            "device_type_": "kDeviceTypeCodeCpu:0",
            "is_dynamic_input_": false,
            "inputs_": [
                {
                    "desc_": "input_0",
                    "name_": "OpenCvImageDecode_22@output_0",
                    "type_": "ndarray"
                }
            ],
            "is_dynamic_output_": false,
            "outputs_": [
                {
                    "desc_": "output_0",
                    "name_": "CvtResizeNormTrans_17@output_0",
                    "type_": "Tensor"
                }
            ],
            "node_type_": "Intermediate",
            "param_": {
                "src_pixel_type_": "kPixelTypeBGR",
                "dst_pixel_type_": "kPixelTypeRGB",
                "interp_type_": "kInterpTypeLinear",
                "h_": 640,
                "w_": 640,
                "data_type_": "kDataTypeCodeFp32",
                "data_format_": "kDataFormatNCHW",
                "normalize_": true,
                "scale_": [
                    0.003921568859368563,
                    0.003921568859368563,
                    0.003921568859368563,
                    0.003921568859368563
                ],
                "mean_": [
                    0,
                    0,
                    0,
                    0
                ],
                "std_": [
                    1,
                    1,
                    1,
                    1
                ]
            },
            "size": {
                "width": 200,
                "height": 80
            },
            "node_repository_": []
        },
        {
            "key_": "nndeploy::infer::Infer",
            "name_": "Infer_18",
            "desc_": "Universal Inference Node - Enables cross-platform model deployment with multiple inference backends while maintaining native performance",
            "device_type_": "kDeviceTypeCodeCpu:0",
            "is_dynamic_input_": true,
            "inputs_": [
                {
                    "desc_": "input_0",
                    "name_": "CvtResizeNormTrans_17@output_0",
                    "type_": "Tensor"
                }
            ],
            "is_dynamic_output_": true,
            "outputs_": [
                {
                    "desc_": "output_0",
                    "name_": "Infer_18@output_0",
                    "type_": "Tensor"
                }
            ],
            "node_type_": "Intermediate",
            "type_": "kInferenceTypeOnnxRuntime",
            "param_": {
                "model_type_": "kModelTypeOnnx",
                "is_path_": true,
                "model_value_": [
                    "resources/models/detect/yolo11s.sim.onnx"
                ],
                "external_model_data_": [
                    ""
                ],
                "device_type_": "kDeviceTypeCodeCpu:0",
                "num_thread_": 8,
                "gpu_tune_kernel_": 1,
                "input_num_": 1,
                "input_name_": [
                    ""
                ],
                "input_shape_": [
                    [
                        -1,
                        -1,
                        -1,
                        -1
                    ]
                ],
                "output_num_": 1,
                "output_name_": [
                    ""
                ],
                "encrypt_type_": "kEncryptTypeNone",
                "license_": "",
                "share_memory_mode_": "kShareMemoryTypeNoShare",
                "precision_type_": "kPrecisionTypeFp32",
                "power_type_": "kPowerTypeNormal",
                "is_dynamic_shape_": false,
                "min_shape_": {
                    "input_0": [
                        -1,
                        -1,
                        -1,
                        -1
                    ]
                },
                "opt_shape_": {
                    "input_0": [
                        -1,
                        -1,
                        -1,
                        -1
                    ]
                },
                "max_shape_": {
                    "input_0": [
                        -1,
                        -1,
                        -1,
                        -1
                    ]
                },
                "parallel_type_": "kParallelTypeNone",
                "worker_num_": 1
            },
            "size": {
                "width": 200,
                "height": 80
            },
            "node_repository_": []
        },
        {
            "key_": "nndeploy::detect::YoloPostProcess",
            "name_": "YoloPostProcess_19",
            "desc_": "YOLO v5/v6/v7/v8/v11 postprocess[device::Tensor->DetectResult]",
            "device_type_": "kDeviceTypeCodeCpu:0",
            "is_dynamic_input_": false,
            "inputs_": [
                {
                    "desc_": "input_0",
                    "name_": "Infer_18@output_0",
                    "type_": "Tensor"
                }
            ],
            "is_dynamic_output_": false,
            "outputs_": [
                {
                    "desc_": "output_0",
                    "name_": "YoloPostProcess_19@output_0",
                    "type_": "DetectResult"
                }
            ],
            "node_type_": "Intermediate",
            "param_": {
                "version_": 11,
                "score_threshold_": 0.5,
                "nms_threshold_": 0.45,
                "num_classes_": 80,
                "model_h_": 640,
                "model_w_": 640
            },
            "size": {
                "width": 200,
                "height": 80
            },
            "node_repository_": []
        },
        {
            "key_": "nndeploy::detect::DrawBox",
            "name_": "DrawBox_20",
            "desc_": "Draw detection boxes on input cv::Mat image based on detection results[cv::Mat->cv::Mat]",
            "device_type_": "kDeviceTypeCodeCpu:0",
            "is_dynamic_input_": false,
            "inputs_": [
                {
                    "desc_": "input_0",
                    "name_": "OpenCvImageDecode_22@output_0",
                    "type_": "ndarray"
                },
                {
                    "desc_": "input_1",
                    "name_": "YoloPostProcess_19@output_0",
                    "type_": "DetectResult"
                }
            ],
            "is_dynamic_output_": false,
            "outputs_": [
                {
                    "desc_": "output_0",
                    "name_": "DrawBox_20@output_0",
                    "type_": "ndarray"
                }
            ],
            "node_type_": "Intermediate",
            "size": {
                "width": 200,
                "height": 80
            },
            "node_repository_": []
        },
        {
            "key_": "nndeploy::codec::OpenCvImageDecode",
            "name_": "OpenCvImageDecode_22",
            "developer_": "",
            "source_": "",
            "desc_": "Decode image using OpenCV, from image path to cv::Mat, default color space is BGR",
            "device_type_": "kDeviceTypeCodeCpu:0",
            "version_": "1.0.0",
            "required_params_": [
                "path_"
            ],
            "ui_params_": [],
            "is_dynamic_input_": false,
            "inputs_": [],
            "is_dynamic_output_": false,
            "outputs_": [
                {
                    "desc_": "output_0",
                    "name_": "OpenCvImageDecode_22@output_0",
                    "type_": "ndarray"
                }
            ],
            "node_type_": "Input",
            "io_type_": "Image",
            "path_": "resources/template/nndeploy-workflow/detect/zidane.jpg",
            "size": {
                "width": 200,
                "height": 80
            },
            "node_repository_": []
        },
        {
            "key_": "nndeploy::codec::OpenCvImageEncode",
            "name_": "OpenCvImageEncode_23",
            "developer_": "",
            "source_": "",
            "desc_": "Encode image using OpenCV, from cv::Mat to image file, supports common image formats",
            "device_type_": "kDeviceTypeCodeCpu:0",
            "version_": "1.0.0",
            "required_params_": [
                "path_"
            ],
            "ui_params_": [],
            "is_dynamic_input_": false,
            "inputs_": [
                {
                    "desc_": "input_0",
                    "name_": "DrawBox_20@output_0",
                    "type_": "ndarray"
                }
            ],
            "is_dynamic_output_": false,
            "outputs_": [],
            "node_type_": "Output",
            "io_type_": "Image",
            "path_": "resources/images/result.yolo.jpg",
            "size": {
                "width": 200,
                "height": 80
            },
            "node_repository_": []
        }
    ],
    "nndeploy_ui_layout": {
        "layout": {
            "CvtResizeNormTrans_17": {
                "position": {
                    "x": 380,
                    "y": 104.4
                },
                "size": {
                    "width": 200,
                    "height": 80
                },
                "expanded": true
            },
            "Infer_18": {
                "position": {
                    "x": 690,
                    "y": 104.40000000000003
                },
                "size": {
                    "width": 200,
                    "height": 80
                },
                "expanded": true
            },
            "YoloPostProcess_19": {
                "position": {
                    "x": 982,
                    "y": 104.40000000000002
                },
                "size": {
                    "width": 200,
                    "height": 80
                },
                "expanded": true
            },
            "DrawBox_20": {
                "position": {
                    "x": 1274,
                    "y": 7.700000000000017
                },
                "size": {
                    "width": 200,
                    "height": 80
                },
                "expanded": true
            },
            "OpenCvImageDecode_22": {
                "position": {
                    "x": 100,
                    "y": 0
                },
                "size": {
                    "width": 200,
                    "height": 80
                },
                "expanded": true
            },
            "OpenCvImageEncode_23": {
                "position": {
                    "x": 1569,
                    "y": 7.700000000000017
                },
                "size": {
                    "width": 200,
                    "height": 80
                },
                "expanded": true
            }
        },
        "groups": []
    }
}
```