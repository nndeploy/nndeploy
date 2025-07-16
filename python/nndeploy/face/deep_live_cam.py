
# Source code from deep_live_cam.py, modified to comply with nndeploy interface specifications

import os
import shutil
from typing import Any
import cv2
import json
import numpy as np
import insightface

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_cluster_centroids(embeddings, max_k=10) -> Any:
    """
    使用K-means聚类算法寻找人脸嵌入向量的最优聚类中心点
    
    该函数通过分析不同聚类数目下的聚类效果，自动确定最优的聚类数量，
    并返回相应的聚类中心点。主要用于将视频中的多个人脸嵌入向量按照
    相似性进行分组，以识别视频中的不同人物。
    
    参数:
        embeddings: 人脸嵌入向量列表，通常是由InsightFace等模型提取的特征向量
        max_k: 最大聚类数目，默认为10，用于限制搜索范围
    
    返回:
        optimal_centroids: 最优聚类中心点的numpy数组
    
    算法原理:
        1. 遍历从1到max_k的所有可能聚类数目
        2. 对每个k值执行K-means聚类，记录惯性值(inertia)和聚类中心
        3. 计算相邻惯性值的差值，差值最大的点表示聚类效果提升最显著
        4. 选择差值最大对应的聚类数目作为最优解
        
    注意:
        - 惯性值表示样本点到其聚类中心的距离平方和，越小表示聚类效果越好
        - 使用肘部法则(Elbow Method)来确定最优聚类数目
        - random_state=0确保结果的可重复性
    """
    inertia = []  # 存储每个k值对应的惯性值
    cluster_centroids = []  # 存储每个k值对应的聚类信息
    K = range(1, max_k+1)  # 聚类数目范围

    # 对每个可能的聚类数目进行K-means聚类
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(embeddings)
        inertia.append(kmeans.inertia_)
        cluster_centroids.append({"k": k, "centroids": kmeans.cluster_centers_})

    # 计算相邻惯性值的差值，用于寻找肘部点
    diffs = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
    # 找到差值最大的位置，对应最优的聚类数目
    optimal_centroids = cluster_centroids[diffs.index(max(diffs)) + 1]['centroids']

    return optimal_centroids
  

def find_closest_centroid(centroids: list, normed_face_embedding) -> tuple:
    """
    寻找与给定人脸嵌入向量最相似的聚类中心点
    
    该函数通过计算余弦相似度来确定输入的人脸嵌入向量与哪个聚类中心最匹配。
    这在人脸识别和分类任务中非常有用，可以将新的人脸嵌入向量分配到最相似的
    已知人物类别中。
    
    参数:
        centroids: 聚类中心点列表，每个元素都是一个代表某个人物特征的嵌入向量
        normed_face_embedding: 已归一化的人脸嵌入向量，需要找到其最匹配的聚类中心
    
    返回:
        tuple: 包含两个元素的元组
            - closest_centroid_index: 最相似聚类中心的索引位置
            - centroids[closest_centroid_index]: 最相似的聚类中心向量
        或者返回None: 当处理过程中发生ValueError异常时
    
    算法原理:
        1. 将输入的聚类中心列表和人脸嵌入向量转换为numpy数组，便于数值计算
        2. 使用点积运算计算输入向量与所有聚类中心的余弦相似度
        3. 由于输入向量已经归一化，点积结果直接表示余弦相似度
        4. 使用argmax找到相似度最高的聚类中心索引
        5. 返回该索引及对应的聚类中心向量
    
    注意事项:
        - 假设输入的人脸嵌入向量已经进行了L2归一化
        - 使用余弦相似度作为相似性度量标准
        - 包含异常处理机制，确保函数稳定性
    """
    try:
        # 将聚类中心列表转换为numpy数组，便于进行矩阵运算
        centroids = np.array(centroids)
        # 将人脸嵌入向量转换为numpy数组
        normed_face_embedding = np.array(normed_face_embedding)
        
        # 计算输入向量与所有聚类中心的点积，得到相似度分数
        # 由于向量已归一化，点积等价于余弦相似度
        similarities = np.dot(centroids, normed_face_embedding)
        
        # 找到相似度最高的聚类中心索引
        closest_centroid_index = np.argmax(similarities)
        
        # 返回最匹配的聚类中心索引和对应的中心向量
        return closest_centroid_index, centroids[closest_centroid_index]
    except ValueError:
        # 当输入数据格式不正确或计算过程中出现错误时返回None
        return None
      

def create_lower_mouth_mask(
    face, frame: np.ndarray, mask_down_size: float, mask_size
) -> tuple[np.ndarray, np.ndarray, tuple, np.ndarray]:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        #                  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
        lower_lip_order = [
            65,
            66,
            62,
            70,
            69,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            0,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            65,
        ]
        lower_lip_landmarks = landmarks[lower_lip_order].astype(
            np.float32
        )  # Use float for precise calculations

        # Calculate the center of the landmarks
        center = np.mean(lower_lip_landmarks, axis=0)

        # Expand the landmarks outward
        expansion_factor = (
            1 + mask_down_size
        )  # Adjust this for more or less expansion
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center

        # Extend the top lip part
        toplip_indices = [
            20,
            0,
            1,
            2,
            3,
            4,
            5,
        ]  # Indices for landmarks 2, 65, 66, 62, 70, 69, 18
        toplip_extension = (
            mask_size * 0.5
        )  # Adjust this factor to control the extension
        for idx in toplip_indices:
            direction = expanded_landmarks[idx] - center
            direction = direction / np.linalg.norm(direction)
            expanded_landmarks[idx] += direction * toplip_extension

        # Extend the bottom part (chin area)
        chin_indices = [
            11,
            12,
            13,
            14,
            15,
            16,
        ]  # Indices for landmarks 21, 22, 23, 24, 0, 8
        chin_extension = 2 * 0.2  # Adjust this factor to control the extension
        for idx in chin_indices:
            expanded_landmarks[idx][1] += (
                expanded_landmarks[idx][1] - center[1]
            ) * chin_extension

        # Convert back to integer coordinates
        expanded_landmarks = expanded_landmarks.astype(np.int32)

        # Calculate bounding box for the expanded lower mouth
        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        # Add some padding to the bounding box
        padding = int((max_x - min_x) * 0.1)  # 10% padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)

        # Ensure the bounding box dimensions are valid
        if max_x <= min_x or max_y <= min_y:
            if (max_x - min_x) <= 1:
                max_x = min_x + 1
            if (max_y - min_y) <= 1:
                max_y = min_y + 1

        # Create the mask
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [expanded_landmarks - [min_x, min_y]], 255)

        # Apply Gaussian blur to soften the mask edges
        mask_roi = cv2.GaussianBlur(mask_roi, (15, 15), 5)

        # Place the mask ROI in the full-sized mask
        mask[min_y:max_y, min_x:max_x] = mask_roi

        # Extract the masked area from the frame
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

        # Return the expanded lower lip polygon in original frame coordinates
        lower_lip_polygon = expanded_landmarks

    return mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon


def draw_mouth_mask_visualization(
    frame: np.ndarray, face, mouth_mask_data: tuple, mask_feather_ratio: int
) -> np.ndarray:
    landmarks = face.landmark_2d_106
    if landmarks is not None and mouth_mask_data is not None:
        mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = (
            mouth_mask_data
        )

        vis_frame = frame.copy()

        # Ensure coordinates are within frame bounds
        height, width = vis_frame.shape[:2]
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(width, max_x), min(height, max_y)

        # Adjust mask to match the region size
        mask_region = mask[0 : max_y - min_y, 0 : max_x - min_x]

        # Remove the color mask overlay
        # color_mask = cv2.applyColorMap((mask_region * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Ensure shapes match before blending
        vis_region = vis_frame[min_y:max_y, min_x:max_x]
        # Remove blending with color_mask
        # if vis_region.shape[:2] == color_mask.shape[:2]:
        #     blended = cv2.addWeighted(vis_region, 0.7, color_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended

        # Draw the lower lip polygon
        cv2.polylines(vis_frame, [lower_lip_polygon], True, (0, 255, 0), 2)

        # Remove the red box
        # cv2.rectangle(vis_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        # Visualize the feathered mask
        feather_amount = max(
            1,
            min(
                30,
                (max_x - min_x) // mask_feather_ratio,
                (max_y - min_y) // mask_feather_ratio,
            ),
        )
        # Ensure kernel size is odd
        kernel_size = 2 * feather_amount + 1
        feathered_mask = cv2.GaussianBlur(
            mask_region.astype(float), (kernel_size, kernel_size), 0
        )
        feathered_mask = (feathered_mask / feathered_mask.max() * 255).astype(np.uint8)
        # Remove the feathered mask color overlay
        # color_feathered_mask = cv2.applyColorMap(feathered_mask, cv2.COLORMAP_VIRIDIS)

        # Ensure shapes match before blending feathered mask
        # if vis_region.shape == color_feathered_mask.shape:
        #     blended_feathered = cv2.addWeighted(vis_region, 0.7, color_feathered_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended_feathered

        # Add labels
        cv2.putText(
            vis_frame,
            "Lower Mouth Mask",
            (min_x, min_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            vis_frame,
            "Feathered Mask",
            (min_x, max_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return vis_frame
    return frame


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    face_mask: np.ndarray,
    mouth_polygon: np.ndarray,
    mask_feather_ratio: int
) -> np.ndarray:
    min_x, min_y, max_x, max_y = mouth_box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        mouth_cutout is None
        or box_width is None
        or box_height is None
        or face_mask is None
        or mouth_polygon is None
    ):
        return frame

    try:
        resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_mouth_cutout.shape:
            resized_mouth_cutout = cv2.resize(
                resized_mouth_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)

        # Use the provided mouth polygon to create the mask
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        # Apply feathering to the polygon mask
        feather_amount = min(
            30,
            box_width // mask_feather_ratio,
            box_height // mask_feather_ratio,
        )
        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(float), (0, 0), feather_amount
        )
        feathered_mask = feathered_mask / feathered_mask.max()

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi / 255.0)

        combined_mask = combined_mask[:, :, np.newaxis]
        blended = (
            color_corrected_mouth * combined_mask + roi * (1 - combined_mask)
        ).astype(np.uint8)

        # Apply face mask to blended result
        face_mask_3channel = (
            np.repeat(face_mask_roi[:, :, np.newaxis], 3, axis=2) / 255.0
        )
        final_blend = blended * face_mask_3channel + roi * (1 - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception as e:
        pass

    return frame


def create_face_mask(face, frame: np.ndarray) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Convert landmarks to int32
        landmarks = landmarks.astype(np.int32)

        # Extract facial features
        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        right_eye = landmarks[33:42]
        right_eye_brow = landmarks[43:51]
        left_eye = landmarks[87:96]
        left_eye_brow = landmarks[97:105]

        # Calculate forehead extension
        right_eyebrow_top = np.min(right_eye_brow[:, 1])
        left_eyebrow_top = np.min(left_eye_brow[:, 1])
        eyebrow_top = min(right_eyebrow_top, left_eyebrow_top)

        face_top = np.min([right_side_face[0, 1], left_side_face[-1, 1]])
        forehead_height = face_top - eyebrow_top
        extended_forehead_height = int(forehead_height * 5.0)  # Extend by 50%

        # Create forehead points
        forehead_left = right_side_face[0].copy()
        forehead_right = left_side_face[-1].copy()
        forehead_left[1] -= extended_forehead_height
        forehead_right[1] -= extended_forehead_height

        # Combine all points to create the face outline
        face_outline = np.vstack(
            [
                [forehead_left],
                right_side_face,
                left_side_face[
                    ::-1
                ],  # Reverse left side to create a continuous outline
                [forehead_right],
            ]
        )

        # Calculate padding
        padding = int(
            np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
        )  # 5% of face width

        # Create a slightly larger convex hull for padding
        hull = cv2.convexHull(face_outline)
        hull_padded = []
        for point in hull:
            x, y = point[0]
            center = np.mean(face_outline, axis=0)
            direction = np.array([x, y]) - center
            direction = direction / np.linalg.norm(direction)
            padded_point = np.array([x, y]) + direction * padding
            hull_padded.append(padded_point)

        hull_padded = np.array(hull_padded, dtype=np.int32)

        # Fill the padded convex hull
        cv2.fillConvexPoly(mask, hull_padded, 255)

        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (5, 5), 3)

    return mask


def apply_color_transfer(source, target):
    """
    Apply color transfer from target to source image
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    # Reshape mean and std to be broadcastable
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # Perform the color transfer
    source = (source - source_mean) * (target_std / source_std) + target_mean

    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)