import numpy as np

def histogram_data_distribution(data: np.ndarray, num_bins: int = 2048) -> np.ndarray:
    """统计数据分布并放入指定数量的箱子中
    
    Args:
        data: 输入的numpy数组，表示需要统计的数据
        num_bins: 箱子的数量，默认为2048
        
    Returns:
        np.ndarray: 每个箱子的频数
    """
    # 计算直方图
    histogram, bin_edges = np.histogram(data, bins=num_bins)
    
    return histogram

def cosine_similarity(array1: np.ndarray, array2: np.ndarray) -> float:
    """计算两个numpy数组的余弦相似度
    
    余弦相似度用于衡量两个向量的相似程度，通过计算向量间夹角的余弦值得到。
    结果范围在[-1,1]之间:
    1: 表示两个向量方向完全相同，即最相似
    0: 表示两个向量正交，即没有相似性
    -1: 表示两个向量方向完全相反，即最不相似
    
    计算公式为: cos(θ) = (A·B)/(||A||·||B||)
    其中A·B为向量点积, ||A||和||B||分别为向量的模长
    
    Args:
        array1: 第一个numpy数组
        array2: 第二个numpy数组
        
    Returns:
        float: 余弦相似度值,范围在[-1,1]之间
    """
    # 将数组展平为一维
    array1_flat = array1.flatten()
    array2_flat = array2.flatten()
    
    # 计算点积
    dot_product = np.dot(array1_flat, array2_flat)
    
    # 计算模长
    norm1 = np.linalg.norm(array1_flat)
    norm2 = np.linalg.norm(array2_flat)
    
    # 避免除零错误
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    
    return similarity

def mse_similarity(array1: np.ndarray, array2: np.ndarray) -> float:
    """计算两个numpy数组的均方误差(MSE)相似度
    
    均方误差用于衡量两个数组之间的差异程度。
    MSE值越小表示两个数组越相似。
    为了与余弦相似度保持一致的相似度表示:
    - 使用exp(-MSE)将结果映射到(0,1]区间
    - 1表示完全相同(MSE=0)
    - 接近0表示差异很大(MSE很大)
    
    计算公式为: MSE = mean((A-B)^2)
    相似度 = exp(-MSE)
    
    Args:
        array1: 第一个numpy数组
        array2: 第二个numpy数组
        
    Returns:
        float: MSE相似度值,范围在(0,1]之间
    """
    # 将数组展平为一维
    array1_flat = array1.flatten()
    array2_flat = array2.flatten()
    
    # 计算MSE
    mse = np.mean(np.square(array1_flat - array2_flat))
    
    # 将MSE转换为相似度分数
    similarity = np.exp(-mse)
    
    return similarity

def euclidean_similarity(array1: np.ndarray, array2: np.ndarray) -> float:
    """计算两个numpy数组的欧式距离相似度
    
    欧式距离用于衡量两个数组在欧几里得空间中的距离。
    为了与其他相似度保持一致:
    - 使用exp(-distance)将结果映射到(0,1]区间
    - 1表示完全相同(距离为0) 
    - 接近0表示差异很大(距离很大)
    
    计算公式为: distance = sqrt(sum((A-B)^2))
    相似度 = exp(-distance)
    
    Args:
        array1: 第一个numpy数组
        array2: 第二个numpy数组
        
    Returns:
        float: 欧式距离相似度值,范围在(0,1]之间
    """
    # 将数组展平为一维
    array1_flat = array1.flatten()
    array2_flat = array2.flatten()
    
    # 计算欧式距离
    distance = np.sqrt(np.sum(np.square(array1_flat - array2_flat)))
    
    # 将距离转换为相似度分数
    similarity = np.exp(-distance)
    
    return similarity





