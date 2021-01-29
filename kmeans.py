import numpy as np
from PIL import Image  # 加载PIL包，用于加载创建图片
from sklearn.cluster import KMeans  # 加载Kmeans算法
import matplotlib.pyplot as plt  # 绘制图像
from skimage import filters
from skimage.morphology import disk
import os
from skimage import util


def rgb(img):
    """kmeans: R, G, B
    """
    h, w = img.shape[0], img.shape[1]
    dim = 1
    if len(img.shape) == 3:
        dim = 3

    # 加载 Kmeans 聚类算法
    km = KMeans(n_clusters=3)

    # 聚类获取每个像素所属的类别
    label = km.fit_predict(img.reshape(-1, dim))
    label = label.reshape(h, w)

    # 创建一张新的灰度图保存聚类后的结果
    img_seg = label.astype(np.uint8)

    # # 展示
    # plt.imshow(img_seg, plt.cm.gray)
    # plt.show()
    return img_seg


def rgbxy(img):
    """kmeans: R, G, B, x, y
    """
    # img = Image.open(file_path).convert('RGB')
    # img = np.array(img)

    h, w = img.shape[0], img.shape[1]
    if len(img.shape) == 3:
        dim = 3
    else:
        dim = 1
        img = img[:, :, np.newaxis]


    xy_img = np.zeros((h, w, 2), dtype=np.uint8)
    for i in range(h):
        xy_img[i, :, 0] = i
    for i in range(w):
        xy_img[:, i, 1] = i
    # print(xy_img)

    new_img = np.concatenate((img, xy_img), axis=2)

    # 加载 Kmeans 聚类算法
    km = KMeans(n_clusters=3)

    # 聚类获取每个像素所属的类别
    label = km.fit_predict(new_img.reshape(-1, dim+2))
    label = label.reshape(h, w)

    # 创建一张新的灰度图保存聚类后的结果
    img_seg = label.astype(np.uint8)

    # # 展示
    # plt.imshow(img_seg, plt.cm.gray)
    # plt.show()
    return img_seg


def filter_cluster(img):
    # img = Image.open(file_path).convert('L')
    # img = np.array(img)

    img = filters.median(img, disk(5))

    h, w = img.shape[0], img.shape[1]

    # 加载 Kmeans 聚类算法
    km = KMeans(n_clusters=3)

    # 聚类获取每个像素所属的类别
    label = km.fit_predict(img.reshape(-1, 1))
    label = label.reshape(h, w)

    # 创建一张新的灰度图保存聚类后的结果
    img_seg = label.astype(np.uint8)

    # # 展示
    # plt.imshow(img_seg, plt.cm.gray)
    # plt.show()
    return img_seg


if __name__ == '__main__':
    file_list = ['lena.png', 'lena_noise.jpg', 'cameraman.jpg', 'dog.png', 'coins.png', 'yellowlily.jpg', '000011.jpg']
    file_path = os.path.join('./data', file_list[2])
    orig_img = Image.open(file_path).convert('L')
    orig_img = np.array(orig_img)
    # orig_img = util.random_noise(orig_img,mode='s&p')

    rgb_rst = rgb(orig_img.copy())
    rgbxy_rst = rgbxy(orig_img.copy())
    filter_rst = filter_cluster(orig_img.copy())

    plt.subplot(2, 2, 1)
    plt.title('origin')
    plt.imshow(orig_img)

    plt.subplot(2, 2, 2)
    plt.title('kmeans rgb')
    plt.imshow(rgb_rst)

    plt.subplot(2, 2, 3)
    plt.title('kmeans rgbxy')
    plt.imshow(rgbxy_rst)

    plt.subplot(2, 2, 4)
    plt.title('mid filter')
    plt.imshow(filter_rst)

    plt.show()