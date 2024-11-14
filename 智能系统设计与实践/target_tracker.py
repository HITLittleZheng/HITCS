"""
空间暗弱目标检测

算法步骤如下：
- 通过叠加多张图像，计算出背景，然后对剩余图像进行差分，得到目标图像；
- 对首张图像进行目标检测，取面积和亮度最高的目标作为追踪对象；
- 在剩余图像上进行目标追踪，得到目标轨迹。
"""

import os
from argparse import ArgumentParser
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from glob import glob
from heapq import heappop, heappush
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def image_mean(
    images: List[cv2.typing.MatLike], executor: Executor
) -> cv2.typing.MatLike:
    """
    Returns the mean of images.

    :param images: a list of images
    :type images: List[cv2.typing.MatLike]
    :param executor: the executor to use
    :type executor: Executor
    :return: the mean of images
    :rtype: cv2.typing.MatLike
    """
    n = len(images)
    assert n > 0, "images is empty"

    while len(images) > 1:
        remains = images[-1] if len(images) & 1 else None
        images = list(
            executor.map(
                lambda pair: pair[0].astype(np.uint32) + pair[1].astype(np.uint32),
                zip(images[::2], images[1::2]),
            )
        )
        if remains is not None:
            images.append(remains)
    (total,) = images
    return (total / n).astype(np.uint8)


def get_target_coordinates(
    image: cv2.typing.MatLike,
    *,
    n: int = 1,
    min_area: int = 8,
    brightness_threshold: int = 30,
) -> List[Tuple[int, int, int, int]]:
    """
    Returns the coordinates of the target in the image.

    :param image: an image without background
    :type image: cv2.typing.MatLike
    :param n: the number of targets, defaults to 1
    :type n: int, optional
    :param min_area: minimum target area, defaults to 8
    :type min_area: int, optional
    :param brightness_threshold: minimum target brightness, defaults to 30
    :type brightness_threshold: int, optional
    :return: the coordinates of the target in the image
    :rtype: List[Tuple[int, int, int, int]]
    """

    assert image is not None
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target_coordinates: List[Tuple[float, int, int, int, int]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            # print(f"Contour area too small ({area})")
            continue
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y : y + h, x : x + w]
        mean_brightness = cv2.mean(roi)[0]
        if mean_brightness >= brightness_threshold:
            heappush(target_coordinates, (-mean_brightness * area, x, y, w, h))
    return [
        heappop(target_coordinates)[1:] for _ in range(min(n, len(target_coordinates)))
    ]


def track_targets(
    images: List[cv2.typing.MatLike], initial_coords: List[Tuple[int, int, int, int]]
) -> List[np.ndarray]:
    """
    Tracks targets in a list of images using OpenCV's built-in tracker

    :param images: a list of images
    :type images: List[cv2.typing.MatLike]
    :param initial_coords: the initial coordinates of the targets
    :type initial_coords: List[Tuple[int, int, int, int]]
    :return: a list of tracked points
    :rtype: List[np.ndarray]
    """
    assert images, "Images must be provided"
    assert initial_coords, "Initial coordinates must be provided"
    trackers: List[cv2.Tracker] = []

    for coords in initial_coords:
        tracker = cv2.TrackerCSRT.create()
        tracker.init(images[0], coords)
        trackers.append(tracker)

    old_bboxes: np.ndarray = np.array(initial_coords)
    points: List[np.ndarray] = [
        (old_bboxes[:, :2] + old_bboxes[:, 2:] / 2).astype(np.int32)
    ]

    for i, frame in enumerate(images[1:], start=1):
        new_bboxes = []

        for tracker, old_bbox in zip(trackers, old_bboxes):
            success, bbox = tracker.update(frame)
            if success:
                new_bboxes.append(bbox)
            else:
                print(f"tracker {i} lost")
                # reset tracker
                tracker.init(images[i - 1], old_bbox)
                new_bboxes.append(old_bbox)

        new_bboxes = np.array(new_bboxes)
        points.append((new_bboxes[:, :2] + new_bboxes[:, 2:] / 2).astype(np.int32))
        old_bboxes = new_bboxes
    return points


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_dir", help="input directory", nargs="?", default="images/")
    parser.add_argument("-n", "--number", help="number of targets", type=int, default=1)
    parser.add_argument("-s", "--background-sample", help="background sample count", type=int, default=32)
    parser.add_argument("-a", "--min-area", help="minimum target area", type=int, default=4)
    parser.add_argument("-b", "--brightness-threshold", help="minimum target brightness", type=int, default=15)
    ns = parser.parse_args()

    os.chdir(os.path.dirname(__file__))

    with ThreadPoolExecutor() as executor:
        # read images
        raw_images: List[cv2.typing.MatLike] = list(
            executor.map(
                partial(cv2.imread, flags=cv2.IMREAD_GRAYSCALE),
                glob(os.path.join(ns.input_dir, "*.jpg")),
            )
        )
        # calculate background
        background = image_mean(raw_images[::len(raw_images) // ns.background_sample], executor)
        # remove background from images
        images = list(executor.map(partial(cv2.absdiff, background), raw_images))

    # fetch initial coordinates
    for i, image in enumerate(images):
        initial_coords = get_target_coordinates(
            image, n=ns.number, min_area=ns.min_area, brightness_threshold=ns.brightness_threshold
        )
        if not initial_coords:
            print(f"No target found in image {i}")
            continue
        print(f"Initial Target Coordinates: {initial_coords}")
        break
    else:
        raise RuntimeError("No target found")

    # track targets
    points = track_targets(images[i:], initial_coords)

    # show results
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    for p1s, p2s in zip(points, points[1:]):
        for p1, p2 in zip(p1s, p2s):
            cv2.line(background, p1, p2, (0, 0, 255), 3)

    plt.imshow(background)
    plt.axis('off')
    plt.title('Tracking Results')
    plt.show()
