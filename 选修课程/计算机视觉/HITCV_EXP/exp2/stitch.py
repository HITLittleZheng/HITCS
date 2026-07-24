import sys
import cv2
import numpy as np

def compute_homography(src_pts, dst_pts):
    
    """ 请在此处开始编写你的代码：
    
    计算单应矩阵 H。
    
    :param src_pts: 源点坐标
    :param dst_pts: 目标点坐标
    :return: 单应矩阵 H

   将源点坐标(src_pts)变换到目标点坐标(dst_pts)：

    1. 根据输入的源点坐标和目标点坐标,构建一个线性方程组。每对对应的源点和目标点可以提供两个方程。

       其中(x, y)是源点坐标,(u, v)是目标点坐标,h11到h33是单应矩阵H的9个元素。

    2. 将所有点对产生的方程系数存储在矩阵A中。A的每一行对应一个方程的系数。

    3. 使用奇异值分解(SVD)求解方程组Ah=0的解,其中h是单应矩阵H的9个元素组成的列向量。SVD分解得到的矩阵V的最后一列对应最小奇异值,即为方程组的解。

    4. 将解向量重塑为3x3矩阵,并将其除以H[3,3]进行归一化处理,得到最终的单应矩阵H。

    总的来说,通过构建线性方程组并使用SVD求解,得到了将源点变换到目标点的单应矩阵H。单应矩阵H可用于对图像进行变换,实现图像配准、拼接等操作。
    
    """
    num_points = src_pts.shape[0]
    A = np.zeros((2 * num_points, 9)) # 初始化线性方程组矩阵
    
    for i in range(num_points):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u] # 对应于 x 的变换方程
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v] # 对应于 y 的变换方程
        
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3) # 取最小奇异值对应的奇异向量，重塑为3x3矩阵
    H = H / H[2, 2]
    
    return H

def ransac_homography(src_pts, dst_pts, threshold=5.0, max_iter=2000, confidence=0.99):
    
    """ 请在此处开始编写你的代码：
    
    使用RANSAC估计单应矩阵。
    
    :param src_pts: 源点坐标
    :param dst_pts: 目标点坐标
    :param threshold: 内点距离阈值
    :param max_iter: 最大迭代次数
    :param confidence: 置信度
    :return: 最佳单应矩阵 H 和内点掩码

    这个代码模块使用RANSAC算法估计单应矩阵,其思路如下:

    1. RANSAC算法通过反复随机抽样和估计模型参数,从包含异常值的数据中求解最优模型。

    2. 在每次迭代中,随机选择4对源点和目标点,用这4对点计算出一个单应矩阵H。

    3. 使用计算出的单应矩阵H将所有源点变换到目标点的坐标系,并计算变换后的源点与实际目标点之间的距离(残差)。

    4. 根据残差和给定的阈值,判断每个点是否为内点(即符合当前估计的单应矩阵的点)。同时更新最佳单应矩阵和对应的内点数量。

    5. 重复步骤2-4,直到达到最大迭代次数或者内点数量占总点数的比例超过给定的置信度。最终返回最佳单应矩阵和内点掩码。

    通过RANSAC算法,可以在存在噪声和异常值的情况下,鲁棒地估计出最优的单应矩阵,用于图像配准和拼接任务。
    
    """
    num_pts = src_pts.shape[0]
    best_inliers = None
    best_H = None
    iteration = 0
    best_inlier_count = 0
    
    while iteration < max_iter:
        indices = np.random.choice(num_pts, 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]
        
        H_candidate = compute_homography(src_sample, dst_sample)
        
        src_homog = np.column_stack((src_pts, np.ones(num_pts))) #  将 src_pts 的 x,y 坐标和一个全1的列向量堆叠起来
        dst_homog_pred = np.dot(H_candidate, src_homog.T).T
        dst_homog_pred /= dst_homog_pred[:, 2][:, np.newaxis]  # 将变换后的齐次坐标 𝑥′,𝑦′,𝑤′ 归一化，使得每个点的第三个坐标为1
        
        # 计算变换后的点与实际目标点之间的欧几里得距离
        errors = np.sqrt((dst_homog_pred[:, 0] - dst_pts[:, 0]) ** 2 + (dst_homog_pred[:, 1] - dst_pts[:, 1]) ** 2)
        inliers = errors < threshold
        inlier_count = np.sum(inliers)
        
        if inlier_count > best_inlier_count:
            best_inliers = inliers
            best_H = H_candidate
            best_inlier_count = inlier_count
            
            # Update the number of iterations needed for a given confidence level
            inlier_ratio = best_inlier_count / num_pts  # 内点比例
            n_estimated = np.log(1 - confidence) / np.log(1 - inlier_ratio ** 4)
            max_iter = min(max_iter, int(n_estimated))
        
        iteration += 1

    if best_H is not None:
        return best_H, best_inliers.astype(int)
    else:
        return None, None

def Panorama_stitching(image_right, image_left, downscale_factor=2):
    """
    全景拼接函数。
    
    :param image_right: 右侧图像
    :param image_left: 左侧图像
    :param downscale_factor: 下采样因子
    :return: 拼接后的全景图像
    
    """
    # 对图像进行下采样，以期减少计算量
    image_right_downscaled = cv2.resize(image_right, None, fx=1/downscale_factor, fy=1/downscale_factor)
    image_left_downscaled = cv2.resize(image_left, None, fx=1/downscale_factor, fy=1/downscale_factor)

    # 转换为灰度图像
    gray_right = cv2.cvtColor(image_right_downscaled, cv2.COLOR_BGR2GRAY)
    gray_left = cv2.cvtColor(image_left_downscaled, cv2.COLOR_BGR2GRAY)

    # 使用SIFT特征检测器
    sift = cv2.SIFT_create()
    keypoints_right, descriptors_right = sift.detectAndCompute(gray_right, None)  #接受一个灰度图像和一个掩码
    keypoints_left, descriptors_left = sift.detectAndCompute(gray_left, None)
    # 每个关键点都是一个带有许多属性的对象（如位置、尺度、方向等）
    # 一个数组，每行对应一个关键点的描述符

    # 使用暴力匹配器进行特征匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_right, descriptors_left, k=2)

    # 通过Lowe's ratio test筛选出好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([keypoints_right[m.queryIdx].pt for m in good_matches]) * downscale_factor
        dst_pts = np.float32([keypoints_left[m.trainIdx].pt for m in good_matches]) * downscale_factor

        # 使用RANSAC计算单应矩阵
        H, mask = ransac_homography(src_pts, dst_pts)
        matches_mask = np.array(mask, dtype=int).tolist()  # 转换为Int类型列表

        # 图像变形——透视变换
        h_right, w_right = image_right.shape[:2]
        h_left, w_left = image_left.shape[:2]
        panorama = cv2.warpPerspective(image_right, H, (w_right + w_left, max(h_right, h_left)))
        panorama[0:h_left, 0:w_left] = image_left

        # 显示单应矩阵
        print("Homography matrix:")
        print(H)

        # 显示匹配点比例
        print(f"Matches ratio: {len(good_matches) / len(matches):.2f}")

        # 显示匹配点
        draw_params = dict(matchColor=(0, 255, 0),  # 在匹配的关键点间画绿线
                           singlePointColor=None,
                           matchesMask=matches_mask,  # 只画内点
                           flags=2)
        img_matches = cv2.drawMatches(image_right_downscaled, keypoints_right, image_left_downscaled, keypoints_left, good_matches, None, **draw_params)

        # cv2.imshow("Matches", img_matches)
        cv2.imwrite('./output/teachermatches.jpg', img_matches)

        return panorama

    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 10))
        return None

# 主程序
if (__name__ == "__main__"):
    # 读取左右图像
    image_right = cv2.imread('./right.jpg')
    image_left = cv2.imread('./left.jpg')

    # 调用全景拼接函数，得到全景图像
    panorama = Panorama_stitching(image_right, image_left)

    # 判断全景图像是否为空，如果不为空则显示并保存全景图像
    if panorama is not None:
        # 显示全景图像
        # cv2.imshow('Panorama', panorama)
        cv2.imwrite('./output/teacherphoto.jpg', panorama)  # 保存全景图像
        # cv2.waitKey(0)
    else:
        print("Panorama stitching failed.")

    # cv2.destroyAllWindows()
    sys.exit(0)
