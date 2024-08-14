
#ifndef _NNDEPLOY_MODEL_DETECT_UTIL_H_
#define _NNDEPLOY_MODEL_DETECT_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/model/detect/result.h"

namespace nndeploy {
namespace model {

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
extern NNDEPLOY_CC_API std::array<float, 4> getOriginBox(
    float xmin, float ymin, float xmax, float ymax, const float *scale_factor,
    float x_offset, float y_offset, int ori_width, int ori_height);

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
extern NNDEPLOY_CC_API std::array<float, 4> getOriginBox(
    const std::array<float, 4> &box, const float *scale_factor, float x_offset,
    float y_offset, int ori_width, int ori_height);

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
base::Status computeNMS(const DetectResult &src, std::vector<int> &keep_idxs,
                        const float iou_threshold);

base::Status FaastNMS(const DetectResult &src, std::vector<int> &keep_idxs,
                        const float iou_threshold);

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_DETECT_COMMON_H_ */
