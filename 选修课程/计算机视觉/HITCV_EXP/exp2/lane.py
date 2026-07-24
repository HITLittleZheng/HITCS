import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 灰度转换
def convert_to_grayscale(image):
    print("灰度转换已完成 10%")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_kernel(size, sigma=1.0):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


# 滤波
def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    print("高斯滤波已完成 20%")
    kernel = gaussian_kernel(kernel_size, sigma)
    output = np.zeros_like(image)

    # 添加边界填充，以便卷积时边缘像素也能正确处理
    pad_height = kernel_size // 2
    pad_width = kernel_size // 2
    padded_image = np.pad(image, [(pad_height, pad_height), (pad_width, pad_width)], mode='constant', constant_values=0)

    # 对图像的每个像素应用高斯核
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(region * kernel)

    return output


# 边缘检测
def sobel_filters(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = np.convolve(image.flatten(), Kx.flatten(), 'same').reshape(image.shape)
    Iy = np.convolve(image.flatten(), Ky.flatten(), 'same').reshape(image.shape)
    G = np.sqrt(Ix**2 + Iy**2)
    Theta = np.arctan2(Iy, Ix)
    return G, Theta


def non_max_suppression(gradient, direction):
    M, N = gradient.shape
    Z = np.zeros((M, N))
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255
                if (0 <= angle[i,j] < 45) or (135 <= angle[i,j] <= 180):
                    q = gradient[i, j+1]
                    r = gradient[i, j-1]
                elif (45 <= angle[i,j] < 135):
                    q = gradient[i+1, j]
                    r = gradient[i-1, j]

                if gradient[i,j] >= q and gradient[i,j] >= r:
                    Z[i,j] = gradient[i,j]
                else:
                    Z[i,j] = 0
            except IndexError:
                pass
    return Z


def threshold(image, low, high):
    strong = 255
    weak = 25
    result = np.zeros_like(image)
    strong_i, strong_j = np.where(image >= high)
    weak_i, weak_j = np.where((image >= low) & (image < high))
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    return result


def detect_edges(image, low_threshold=50, high_threshold=150):
    print("边缘检测已完成 30%")
    gradient, direction = sobel_filters(image)
    non_max_img = non_max_suppression(gradient, direction)
    final_img = threshold(non_max_img, low_threshold, high_threshold)
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    return final_img


# 提取感兴趣区域ROI：一开始尝试定义为图像下半部分的一个三角形区域。
# 改进后，ROI被定义成一个更符合实际车道形状的梯形区域。
def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [
            (int(width * 0.1), height),
            (int(width * 0.9), height),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.45), int(height * 0.6))
        ]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    print("提取感兴趣区域已完成 40%")
    return masked_image


# 霍夫变换检测直线：通过降低 minLineLength （从100降到20）和增加 maxLineGap （从50增加到300）等，
def hough_transform(image, theta_res=1, rho_res=1, threshold=20):
    height, width = image.shape
    max_dist = int(np.hypot(height, width))
    rhos = np.arange(-max_dist, max_dist, rho_res)
    thetas = np.radians(np.arange(-90, 90, theta_res))

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((2 * max_dist, num_thetas), dtype=np.int32)
    y_idxs, x_idxs = np.nonzero(image)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + max_dist)
            accumulator[rho, t_idx] += 1

    lines = []
    for r_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[r_idx, t_idx] > threshold:
                rho = rhos[r_idx]
                theta = thetas[t_idx]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                lines.append([x1, y1, x2, y2])

    return np.array(lines).reshape(-1, 1, 4)


# 线段后处理：对检测到的线段进行后处理，计算左右车道线的平均斜率和截距；
# 并对检测到的线段进行合并，减少了由于噪声和断裂导致的检测不连续性，使检测结果更加稳定和连续。
def average_slope_intercept(image, lines):
    left_lines = []  # (slope, intercept)
    right_lines = []  # (slope, intercept)
    left_weights = []  # length of the line segment
    right_weights = []  # length of the line segment

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2: # 两个端点的 x 坐标相同，表示这是一条垂直线。跳过
                continue  # skip vertical lines
            slope = (y2 - y1) / (x2 - x1) # 斜率
            intercept = y1 - slope * x1 # 截距
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) # 长度
            if slope < 0:  # left lane 左
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:  # right lane 右
                right_lines.append((slope, intercept))
                right_weights.append(length)
    
    # 加权平均的方法来确定每侧车道的代表性直线
    # 每条线的 (slope, intercept) 与其长度作为权重相乘，然后除以所有权重的和
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    print("线段后处理已完成 80%")
    return left_lane, right_lane


def make_line_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))


def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        left_lane, right_lane = average_slope_intercept(image, lines)
        y1 = image.shape[0]
        y2 = y1 * 0.6
        left_line = make_line_points(y1, y2, left_lane)
        right_line = make_line_points(y1, y2, right_lane)
        for line in [left_line, right_line]:
            if line is not None:
                cv2.line(line_image, *line, (0, 255, 0), 10)  # 绿色线条
    print("车道线绘制已完成 90%")
    return line_image


# 处理单帧图像
def process_image(image):
    gray = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    edges = detect_edges(blurred)
    roi = region_of_interest(edges)
    lines = hough_transform(roi)
    line_image = draw_lines(image, lines)
    combined = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    print("单帧图像处理已完成 100%")
    return combined


# 处理视频
def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_image(frame)
        out.write(processed_frame)
        print("视频处理进度: {:.2f}%".format((cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100))
        print()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("视频处理已完成 100%")


if (__name__ == '__main__'):
    # 处理视频
    process_video('./drive.mp4', './output/teachervideo.mp4')
    sys.exit(0)
