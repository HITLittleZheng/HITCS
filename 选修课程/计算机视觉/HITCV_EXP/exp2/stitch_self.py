import cv2
import numpy as np

# -------------------------- 1. 特征提取与匹配 --------------------------
def extract_sift_features(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(desc1, desc2, ratio_thresh=0.7):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches

def draw_matches(img1, kp1, img2, kp2, matches, save_path="matches.jpg"):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(save_path, img_matches)
    print(f"特征匹配图已保存至 {save_path}")

# -------------------------- 2. 图像拼接（透视变换 + 向量化融合）--------------------------
def stitch_images(img_left, img_right, H):
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]

    left_corners = np.float32([[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]])
    right_corners = np.float32([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]])
    transformed_right_corners = cv2.perspectiveTransform(right_corners.reshape(-1, 1, 2), H).reshape(-1, 2)

    all_corners = np.vstack((left_corners, transformed_right_corners))
    min_x, min_y = np.min(all_corners, axis=0)
    max_x, max_y = np.max(all_corners, axis=0)

    shift_x = -min_x
    shift_y = -min_y
    canvas_width = int(np.ceil(max_x + shift_x))
    canvas_height = int(np.ceil(max_y + shift_y))

    T = np.array([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]], dtype=np.float32)
    H_shifted = T @ H

    warped_right = cv2.warpPerspective(img_right, H_shifted, (canvas_width, canvas_height))
    M_left = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
    warped_left = cv2.warpAffine(img_left, M_left, (canvas_width, canvas_height))

    # ---------- 修正：掩码使用布尔类型 ----------
    mask_left = (warped_left[:, :, 0] > 0).astype(bool)   # 布尔掩码
    mask_right = (warped_right[:, :, 0] > 0).astype(bool)
    overlap_mask = mask_left & mask_right

    panorama = np.zeros_like(warped_left, dtype=np.float32)

    left_only = mask_left & ~overlap_mask
    right_only = mask_right & ~overlap_mask
    panorama[left_only] = warped_left[left_only]
    panorama[right_only] = warped_right[right_only]

    if np.any(overlap_mask):
        cols_with_overlap = np.where(overlap_mask.any(axis=0))[0]
        left_bound = cols_with_overlap[0]
        right_bound = cols_with_overlap[-1]
        overlap_width = right_bound - left_bound + 1

        weight_left_cols = np.linspace(1, 0, overlap_width)
        weight_right_cols = 1 - weight_left_cols

        rows, cols = np.where(overlap_mask)
        col_indices = cols - left_bound
        panorama[rows, cols] = (weight_left_cols[col_indices, np.newaxis] * warped_left[rows, cols] +
                                weight_right_cols[col_indices, np.newaxis] * warped_right[rows, cols])

    return panorama.astype(np.uint8)

# -------------------------- 3. 主程序 --------------------------
def main():
    left_img = cv2.imread("left.jpg")
    right_img = cv2.imread("right.jpg")
    if left_img is None or right_img is None:
        print("错误：无法读取图像文件，请确保 left.jpg 和 right.jpg 存在于当前目录下")
        return

    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    kp1, desc1 = extract_sift_features(gray_left)
    kp2, desc2 = extract_sift_features(gray_right)
    print(f"左图特征点数量: {len(kp1)}, 右图特征点数量: {len(kp2)}")

    good_matches = match_features(desc1, desc2, ratio_thresh=0.7)
    print(f"匹配点对数量（ratio test后）: {len(good_matches)}")

    if len(good_matches) < 4:
        print("匹配点不足，无法估计单应矩阵")
        return

    left_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    right_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    draw_matches(left_img, kp1, right_img, kp2, good_matches, "./output_self/matches.jpg")

    H, mask = cv2.findHomography(right_pts, left_pts, method=cv2.RANSAC,
                                 ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99)

    if H is None:
        print("RANSAC未能估计出有效的单应矩阵")
        return

    n_inliers = np.sum(mask)
    print(f"RANSAC内点数量: {n_inliers} / {len(good_matches)} (比例: {n_inliers/len(good_matches):.2f})")

    panorama = stitch_images(left_img, right_img, H)
    cv2.imwrite("./output_self/panorama.jpg", panorama)
    print("全景拼接图已保存至 ./output_self/panorama.jpg")

if __name__ == "__main__":
    main()
