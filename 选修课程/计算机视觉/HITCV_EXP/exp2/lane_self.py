# ============================================================
# lane.py - 视频车道线检测程序（优化版）
# 功能：逐帧处理视频，灰度化、高斯滤波、Canny边缘检测、
#       ROI提取、霍夫直线检测、左右车道线拟合与绘制，
#       输出 output_drive_V3.mp4
# 改进：NumPy向量化处理、cv2.fitLine拟合、动态ROI
# ============================================================

import cv2
import numpy as np

# 可调参数（供实验分析）
GAUSSIAN_KERNEL = (5, 5)      # 高斯滤波核大小
CANNY_THRESH1 = 50            # Canny低阈值
CANNY_THRESH2 = 150           # Canny高阈值
HOUGH_RHO = 1                 # 霍夫变换距离精度
HOUGH_THRESH = 50             # 霍夫累加器阈值
HOUGH_MIN_LINE_LEN = 50       # 最小线段长度
HOUGH_MAX_LINE_GAP = 100      # 线段最大间隔

def region_of_interest(img, vertices):
    """
    提取感兴趣区域，将非ROI区域置黑
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def fit_line_with_cv2(points, y_top, y_bottom):
    """
    使用 cv2.fitLine 拟合直线，并返回直线与 y = y_top 和 y = y_bottom 的交点
    points: Nx2 数组，存储端点坐标
    y_top, y_bottom: 直线交点的 y 坐标（图像顶部和底部区域）
    返回 (x1, y1, x2, y2) 或 None
    """
    if points is None or len(points) < 2:
        return None
    # points 形状 (N,2)，需转为 (N,1,2) 用于 cv2.fitLine
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    # 拟合直线，输出 (vx, vy, x0, y0)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]
    # 防止除零（直线接近水平时）
    if abs(vy) < 1e-6:
        return None
    # 计算与 y = y_top 的交点
    t_top = (y_top - y0) / vy
    x_top = int(x0 + vx * t_top)
    # 计算与 y = y_bottom 的交点
    t_bottom = (y_bottom - y0) / vy
    x_bottom = int(x0 + vx * t_bottom)
    return (x_top, y_top, x_bottom, y_bottom)

def draw_lanes(frame, lines, roi_y_top, roi_y_bottom):
    """
    根据霍夫直线检测结果，分离左右车道线，拟合直线并绘制
    参数 lines: 霍夫变换返回的线段列表，每个元素为 [[x1,y1,x2,y2]]
    roi_y_top, roi_y_bottom: ROI 的上下边界 y 坐标，用于确定绘制的直线范围
    """
    if lines is None:
        return frame
    
    # 将线段列表转换为 NumPy 数组 (N,4)
    line_arr = np.array([line[0] for line in lines], dtype=np.float32)
    if line_arr.shape[0] == 0:
        return frame
    
    # 提取坐标
    x1 = line_arr[:, 0]
    y1 = line_arr[:, 1]
    x2 = line_arr[:, 2]
    y2 = line_arr[:, 3]
    
    # 计算斜率，避免除零
    dx = x2 - x1
    dy = y2 - y1
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = dy / dx
    
    # 过滤条件：斜率绝对值 < 0.3 或 水平线（dy为0）或 dx=0（竖直线）
    valid_mask = (np.abs(slope) >= 0.3) & (dx != 0) & (dy != 0)
    if not np.any(valid_mask):
        return frame
    
    x1, y1, x2, y2 = x1[valid_mask], y1[valid_mask], x2[valid_mask], y2[valid_mask]
    slope = slope[valid_mask]
    
    # 分类左右车道线（左：负斜率，右：正斜率）
    left_mask = slope < 0
    right_mask = slope > 0
    
    left_points = []
    right_points = []
    
    # 收集左车道线的所有端点
    if np.any(left_mask):
        left_x1 = x1[left_mask]
        left_y1 = y1[left_mask]
        left_x2 = x2[left_mask]
        left_y2 = y2[left_mask]
        # 每个线段提供两个端点
        left_points = np.vstack([np.column_stack((left_x1, left_y1)),
                                 np.column_stack((left_x2, left_y2))])
    
    # 收集右车道线的所有端点
    if np.any(right_mask):
        right_x1 = x1[right_mask]
        right_y1 = y1[right_mask]
        right_x2 = x2[right_mask]
        right_y2 = y2[right_mask]
        right_points = np.vstack([np.column_stack((right_x1, right_y1)),
                                  np.column_stack((right_x2, right_y2))])
    
    # 拟合左右车道线
    left_line = fit_line_with_cv2(left_points, roi_y_top, roi_y_bottom) if len(left_points) >= 2 else None
    right_line = fit_line_with_cv2(right_points, roi_y_top, roi_y_bottom) if len(right_points) >= 2 else None
    
    # 绘制车道线
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    return frame

def process_frame(frame, roi_vertices, roi_y_top, roi_y_bottom):
    """
    对单帧图像进行完整车道线检测流程
    返回处理后的帧（带车道线）
    """
    # 1. 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 高斯滤波
    blur = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    
    # 3. Canny边缘检测
    edges = cv2.Canny(blur, CANNY_THRESH1, CANNY_THRESH2)
    
    # 4. ROI提取
    roi_edges = region_of_interest(edges, roi_vertices)
    
    # 5. 霍夫直线检测
    lines = cv2.HoughLinesP(roi_edges, HOUGH_RHO, np.pi/180, HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LINE_LEN,
                            maxLineGap=HOUGH_MAX_LINE_GAP)
    
    # 6. 绘制车道线
    lane_frame = draw_lanes(frame.copy(), lines, roi_y_top, roi_y_bottom)
    
    return lane_frame

def main():
    # 输入输出视频路径
    input_video = "drive.mp4"
    output_video = "./output_self/output_drive_V3.mp4"
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 动态计算 ROI 顶点（适用于 720p 比例，可调整）
    # 底部取整行，顶部取图像高度的 45% 处
    y_top = int(height * 0.6)
    y_bottom = height
    x_offset = width // 4
    roi_vertices = np.array([[(0, y_bottom), (width//2 - x_offset//2, y_top),
                              (width//2 + x_offset//2, y_top), (width, y_bottom)]], dtype=np.int32)
    
    print(f"视频分辨率: {width}x{height}, ROI 上边界 y={y_top}, 下边界 y={y_bottom}")
    
    # 定义视频编码器和输出对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理当前帧
        result_frame = process_frame(frame, roi_vertices, y_top, y_bottom)
        out.write(result_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"车道线检测视频已保存至 {output_video}")

if __name__ == "__main__":
    main()
