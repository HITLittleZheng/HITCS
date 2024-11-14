import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from pathlib import Path

import cv2
import numpy as np


def process_single_image(
    filename: Path,
    background: cv2.typing.MatLike,
    star: cv2.typing.MatLike,
    x: int,
    y: int,
    noise_size: float,
) -> Path:
    img = background.copy()
    img[x : x + 5, y : y + 7] += star[:5, :7]

    # 添加高斯噪声
    noise = noise_size * np.random.randn(*img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    r = np.random.rand()
    if r > 0.7:
        img = (r * img).astype(np.uint8)

    cv2.imwrite(filename, img)  # type: ignore
    return filename


def process_images(
    background: cv2.typing.MatLike,
    star: cv2.typing.MatLike,
    output_dir: Path,
    x0: int = 400,
    y0: int = 400,
    sx: int = 1,
    sy: int = 2,
    noise_size: float = 24.0,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        futures = []

        for k, (x, y) in enumerate(zip(count(x0, sx), count(y0, sy)), start=1):
            if x < 0 or y < 0 or x + 5 >= background.shape[0] or y + 7 >= background.shape[1]:
                break

            futures.append(
                executor.submit(
                    process_single_image,
                    output_dir / f"t{k:04}.jpg",
                    background,
                    star,
                    x,
                    y,
                    noise_size,
                )
            )

        # 等待所有线程完成
        for future in as_completed(futures):
            print(f"Saved: {future.result()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and add noise.')
    parser.add_argument('input_image', type=Path, nargs='?', help='Path to the input image (default: all-star.jpg)', default='all-star.jpg')
    parser.add_argument('input_image0', type=Path, nargs='?', help='Path to the second image (default: i0.jpg)', default='i0.jpg')
    parser.add_argument('output_dir', type=Path, nargs='?', help='Output directory for processed images (default: output_images)', default='../images1')
    parser.add_argument('--x0', type=int, default=400, help='Initial x position for overlay (default: 400)')
    parser.add_argument('--y0', type=int, default=400, help='Initial y position for overlay (default: 400)')
    parser.add_argument('--sx', type=int, default=1, help='Step size in x direction (default: 1)')
    parser.add_argument('--sy', type=int, default=2, help='Step size in y direction (default: 2)')
    parser.add_argument('--noise_size', type=float, default=24.0, help='Size of noise to add (default: 24.0)')

    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    background = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
    star = cv2.imread(args.input_image0, cv2.IMREAD_GRAYSCALE)
    assert background is not None and star is not None, 'Failed to load image'
    
    process_images(background, star, args.output_dir, args.x0, args.y0, args.sx, args.sy, args.noise_size)
