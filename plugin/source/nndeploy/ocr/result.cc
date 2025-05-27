
#include "nndeploy/ocr/result.h"

namespace nndeploy {
namespace ocr {

// 清空所有结果数据
void OCRResult::clear() {
  boxes_.clear();
  text_.clear();
  rec_scores_.clear();
  cls_scores_.clear();
  cls_labels_.clear();
  table_boxes_.clear();
  table_structure_.clear();
  table_html_.clear();
}
/**
 * @brief 获取所有识别文本的连接字符串
 * @details
 * 将OCR识别结果格式化为可读的字符串形式，包括检测框坐标、识别文本、置信度分数等信息
 * @return std::string 格式化后的结果字符串，如果没有结果则返回"No Results!"
 */
std::string OCRResult::getText() {
  // 情况1：有文本检测结果（包含边界框信息）
  if (boxes_.size() > 0) {
    std::string out;
    // 遍历每个检测到的文本区域
    for (int n = 0; n < boxes_.size(); n++) {
      // 输出检测框坐标信息
      out = out + "det boxes: [";
      // 输出四个顶点的坐标：[x1,y1],[x2,y2],[x3,y3],[x4,y4]
      for (int i = 0; i < 4; i++) {
        out = out + "[" + std::to_string(boxes_[n][i * 2]) + "," +
              std::to_string(boxes_[n][i * 2 + 1]) + "]";

        if (i != 3) {
          out = out + ",";
        }
      }
      out = out + "]";

      // 如果有文本识别结果，输出识别的文本内容和置信度
      if (rec_scores_.size() > 0) {
        out = out + "rec text: " + text_[n] +
              " rec score:" + std::to_string(rec_scores_[n]) + " ";
      }
      // 如果有文本方向分类结果，输出分类标签和置信度
      if (cls_labels_.size() > 0) {
        out = out + "cls label: " + std::to_string(cls_labels_[n]) +
              " cls score: " + std::to_string(cls_scores_[n]);
      }
      out = out + "\n";
    }

    // 如果有表格检测和结构识别结果
    if (table_boxes_.size() > 0 && table_structure_.size() > 0) {
      // 输出表格中每个单元格的边界框坐标
      for (int n = 0; n < table_boxes_.size(); n++) {
        out = out + "table boxes: [";
        // 输出表格单元格的四个顶点坐标
        for (int i = 0; i < 4; i++) {
          out = out + "[" + std::to_string(table_boxes_[n][i * 2]) + "," +
                std::to_string(table_boxes_[n][i * 2 + 1]) + "]";

          if (i != 3) {
            out = out + ",";
          }
        }
        out = out + "]\n";
      }

      // 输出表格结构标记（HTML标签）
      out = out + "\ntable structure: \n";
      for (int m = 0; m < table_structure_.size(); m++) {
        out += table_structure_[m];
      }

      // 如果有完整的表格HTML代码，一并输出
      if (!table_html_.empty()) {
        out = out + "\n" + "table html: \n" + table_html_;
      }
    }
    return out;

    // 情况2：没有检测框但有文本识别和方向分类结果
  } else if (boxes_.size() == 0 && rec_scores_.size() > 0 &&
             cls_scores_.size() > 0) {
    std::string out;
    // 输出每个文本的识别结果和方向分类结果
    for (int i = 0; i < rec_scores_.size(); i++) {
      out = out + "rec text: " + text_[i] +
            " rec score:" + std::to_string(rec_scores_[i]) + " ";
      out = out + "cls label: " + std::to_string(cls_labels_[i]) +
            " cls score: " + std::to_string(cls_scores_[i]);
      out = out + "\n";
    }
    return out;

    // 情况3：只有文本方向分类结果
  } else if (boxes_.size() == 0 && rec_scores_.size() == 0 &&
             cls_scores_.size() > 0) {
    std::string out;
    // 输出每个文本的方向分类结果
    for (int i = 0; i < cls_scores_.size(); i++) {
      out = out + "cls label: " + std::to_string(cls_labels_[i]) +
            " cls score: " + std::to_string(cls_scores_[i]);
      out = out + "\n";
    }
    return out;

    // 情况4：只有文本识别结果，没有方向分类
  } else if (boxes_.size() == 0 && rec_scores_.size() > 0 &&
             cls_scores_.size() == 0) {
    std::string out;
    // 输出每个文本的识别结果
    for (int i = 0; i < rec_scores_.size(); i++) {
      out = out + "rec text: " + text_[i] +
            " rec score:" + std::to_string(rec_scores_[i]) + " ";
      out = out + "\n";
    }
    return out;

    // 情况5：只有表格检测和结构识别结果
  } else if (boxes_.size() == 0 && table_boxes_.size() > 0 &&
             table_structure_.size() > 0) {
    std::string out;
    // 输出表格中每个单元格的边界框坐标
    for (int n = 0; n < table_boxes_.size(); n++) {
      out = out + "table boxes: [";
      // 输出表格单元格的四个顶点坐标
      for (int i = 0; i < 4; i++) {
        out = out + "[" + std::to_string(table_boxes_[n][i * 2]) + "," +
              std::to_string(table_boxes_[n][i * 2 + 1]) + "]";

        if (i != 3) {
          out = out + ",";
        }
      }
      out = out + "]\n";
    }

    // 输出表格结构标记
    out = out + "\ntable structure: \n";
    for (int m = 0; m < table_structure_.size(); m++) {
      out += table_structure_[m];
    }

    // 如果有完整的表格HTML代码，一并输出
    if (!table_html_.empty()) {
      out = out + "\n" + "table html: \n" + table_html_;
    }
    return out;
  } else {
    // 默认情况：没有任何识别结果
    return "No Results!";
  }
};

}  // namespace ocr
}  // namespace nndeploy