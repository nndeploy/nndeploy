
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _NNDEPLOY_OCR_RESULT_H_
#define _NNDEPLOY_OCR_RESULT_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace ocr {

/**
 * @brief OCR识别结果类
 * @details
 * 用于存储OCR（光学字符识别）的完整结果，包括文本检测、识别、分类和表格解析等功能的输出
 */
class NNDEPLOY_CC_API OCRResult : public base::Param {
 public:
  OCRResult() {};
  virtual ~OCRResult() {};

  // 文本检测结果：每个检测到的文本区域的边界框坐标
  // 每个数组包含8个整数：[x1, y1, x2, y2, x3, y3, x4, y4] 表示四个顶点坐标
  std::vector<std::array<int, 8>> boxes_;

  // 文本识别结果：识别出的文本内容
  std::vector<std::string> text_;
  // 文本识别置信度分数：每个识别文本的可信度，范围通常为[0, 1]
  std::vector<float> rec_scores_;

  // 文本方向分类置信度分数：判断文本是否需要旋转的置信度
  std::vector<float> cls_scores_;
  // 文本方向分类标签：0表示正向，1表示需要180度旋转
  std::vector<int32_t> cls_labels_;

  // 表格检测结果：表格中每个单元格的边界框坐标
  // 格式与boxes_相同，每个数组包含8个整数表示四个顶点坐标
  std::vector<std::array<int, 8>> table_boxes_;
  // 表格结构识别结果：表格的结构标记，如<td>、<tr>等HTML标签
  std::vector<std::string> table_structure_;
  // 完整的表格HTML代码：将表格内容和结构组合成完整的HTML表格
  std::string table_html_;

  // 清空所有结果数据
  void clear();
  // 获取所有识别文本的连接字符串
  std::string getText();
};

}  // namespace ocr
}  // namespace nndeploy

#endif /* _NNDEPLOY_OCR_RESULT_H_ */
