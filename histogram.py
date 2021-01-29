"""
直方图图像分割
1. 直方图双峰法
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def find_threshold(hist):
    # 寻找两个峰值之间的谷值
    sorted_index = np.argsort(hist)
    ia, ib = sorted_index[-1], sorted_index[-2]
    if ia > ib:
        ia, ib = ib, ia

    min_idx = np.argmin(hist[ia:ib+1])
    min_idx += ia

    return min_idx


if __name__ == '__main__':
    file_path = './data/lena.png'
    img_gray = Image.open(file_path).convert('L')
    img_gray = np.array(img_gray)

    hist, bins = np.histogram(img_gray.flatten(), bins=256, range=(0, 256))
    print(hist)

    # draw histogram
    # plt.hist(img_gray.flatten(), 256, [0, 256])
    # plt.show()

    threshold = find_threshold(hist)

    img_seg = img_gray.copy()
    img_seg[img_seg<threshold] = 0
    img_seg[img_seg>=threshold] = 255
    print(img_seg)
    plt.imshow(img_seg)
    plt.show()