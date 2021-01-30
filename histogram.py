"""
直方图图像分割
1. 直方图双峰法
2. 
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters
from skimage.filters.rank.generic import threshold
from skimage.morphology import disk
from sklearn.cluster import KMeans


def threshold_double_peak(file_path):
    img_gray = Image.open(file_path).convert('L')   
    img_gray = np.array(img_gray)

    hist, bins = np.histogram(img_gray.flatten(), bins=256, range=(0, 256))

    # draw histogram
    plt.hist(img_gray.flatten(), 256, [0, 256])
    plt.show()

    # min_idx = np.argmin(hist[ia:ib+1])
    # min_idx += ia

    # threshold = min_idx
    threshold = 70
    print('threshold = {}'.format(threshold))

    img_seg = img_gray.copy()
    img_seg[img_seg<threshold] = 0
    img_seg[img_seg>=threshold] = 255

    return img_seg


def threshold_cluster(file_path):
    img_gray = Image.open(file_path).convert('L')   
    img_gray = np.array(img_gray)

    hist, bins = np.histogram(img_gray.flatten(), bins=256, range=(0, 256))

    # draw histogram
    plt.hist(img_gray.flatten(), 256, [0, 256])
    plt.show()

    # min_idx = np.argmin(hist[ia:ib+1])
    # min_idx += ia

    # threshold = min_idx
    threshold = 90
    print('threshold = {}'.format(threshold))

    img_seg = img_gray.copy()
    img_seg[img_seg<threshold] = 0
    img_seg[img_seg>=threshold] = 255

    return img_seg


def cluster_2d(file_path):
    img_gray = Image.open(file_path).convert('L')   
    img_gray = np.array(img_gray)
    
    img_filt = filters.median(img_gray, disk(5))

    img_gray = img_gray[:, :, np.newaxis]
    img_filt = img_filt[:, :, np.newaxis]

    img_new = np.concatenate((img_gray, img_filt), axis=2)
    h, w = img_new.shape[0], img_new.shape[1]

    km = KMeans(n_clusters=2)
    label = km.fit_predict(img_new.reshape(-1, 2))
    label = label.reshape(h, w)

    # 创建一张新的灰度图保存聚类后的结果
    img_seg = label.astype(np.uint8)
    
    return img_seg


if __name__ == '__main__':
    file_path = './data/lena.png'
    # file_path = './data/lena_noise.jpg'
    # file_path = './data/coins.png'

    img_gray = Image.open(file_path).convert('L')   
    img_gray = np.array(img_gray)

    img_seg = threshold_double_peak(file_path)
    # img_seg = cluster_2d(file_path)

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(img_gray, plt.cm.gray)

    plt.subplot(1, 2, 2)
    plt.title('seg')
    plt.imshow(img_seg)

    plt.show()