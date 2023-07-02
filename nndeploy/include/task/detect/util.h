
#ifndef _NNDEPLOY_SOURCE_TASK_DETECT_UTIL_H_
#define _NNDEPLOY_SOURCE_TASK_DETECT_UTIL_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/param.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/type.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/mat.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/task/detect/result.h"

namespace nndeploy {
namespace task {

/**
 * @brief Get the Origin Box object
 *
 * @param xmin
 * @param ymin
 * @param xmax
 * @param ymax
 * @param scale_factor
 * @param x_offset
 * @param y_offset
 * @param ori_width
 * @param ori_height
 * @return std::array<float, 4>
 */
std::array<float, 4> getOriginBox(float xmin, float ymin, float xmax,
                                  float ymax, const float *scale_factor,
                                  float x_offset, float y_offset, int ori_width,
                                  int ori_height);

/**
 * @brief Get the Origin Box object
 *
 * @param box
 * @param scale_factor
 * @param x_offset
 * @param y_offset
 * @param ori_width
 * @param ori_height
 * @return * std::array<float, 4>
 */
std::array<float, 4> getOriginBox(const std::array<float, 4> &box,
                                  const float *scale_factor, float x_offset,
                                  float y_offset, int ori_width,
                                  int ori_height);

/**
 * @brief
 *
 * @param xmin0
 * @param ymin0
 * @param xmax0
 * @param ymax0
 * @param xmin1
 * @param ymin1
 * @param xmax1
 * @param ymax1
 * @return float
 */
float computeIOU(float xmin0, float ymin0, float xmax0, float ymax0,
                 float xmin1, float ymin1, float xmax1, float ymax1);

/**
 * @brief
 *
 * @param box0 [xmin0, ymin0, xmax0, ymax0]
 * @param box1 [xmin1, ymin1, xmax1, ymax1]
 * @return * float
 */
float computeIOU(const std::array<float, 4> &box0,
                 const std::array<float, 4> &box1);

/**
 * @brief
 *
 * @param boxes [xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1]
 * @param i
 * @param j
 * @return float
 */
float computeIOU(const float *boxes, int i, int j);

/**
 * @brief
 *
 * @param src
 * @param keep_idxs
 * @param iou_threshold
 * @return base::Status
 */
base::Status computeNMS(const DetectResults &src, std::vector<int> &keep_idxs,
                        const float iou_threshold);

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_TASK_DETECT_COMMON_H_ */
