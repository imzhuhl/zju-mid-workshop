import numpy as np
from PIL import Image  # 加载PIL包，用于加载创建图片
from sklearn.cluster import KMeans  # 加载Kmeans算法
import matplotlib.pyplot as plt  # 绘制图像
from skimage import filters
from skimage.morphology import disk

def rgb(file_path):
    """kmeans: R, G, B
    """
    img = Image.open(file_path).convert('RGB')
    img = np.array(img)

    h, w = img.shape[0], img.shape[1]

    # 加载 Kmeans 聚类算法
    km = KMeans(n_clusters=2)

    # 聚类获取每个像素所属的类别
    label = km.fit_predict(img.reshape(-1, 3))
    label = label.reshape(h, w)

    # 创建一张新的灰度图保存聚类后的结果
    img_seg = label.astype(np.uint8)

    # # 展示
    # plt.imshow(img_seg, plt.cm.gray)
    # plt.show()
    return img_seg


def rgbxy(file_path):
    """kmeans: R, G, B, x, y
    """
    img = Image.open(file_path).convert('RGB')
    img = np.array(img)

    h, w = img.shape[0], img.shape[1]

    xy_img = np.zeros((h, w, 2), dtype=np.uint8)
    for i in range(h):
        xy_img[i, :, 0] = i
    for i in range(w):
        xy_img[:, i, 1] = i
    # print(xy_img)

    new_img = np.concatenate((img, xy_img), axis=2)

    # 加载 Kmeans 聚类算法
    km = KMeans(n_clusters=2)

    # 聚类获取每个像素所属的类别
    label = km.fit_predict(new_img.reshape(-1, 5))
    label = label.reshape(h, w)

    # 创建一张新的灰度图保存聚类后的结果
    img_seg = label.astype(np.uint8)

    # # 展示
    # plt.imshow(img_seg, plt.cm.gray)
    # plt.show()
    return img_seg


def filter_cluster(file_paht):
    img = Image.open(file_path).convert('L')
    img = np.array(img)

    img = filters.median(img,disk(5))

    h, w = img.shape[0], img.shape[1]

    # 加载 Kmeans 聚类算法
    km = KMeans(n_clusters=2)

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
    file_list = ['lena.png', 'lena_noise.jpg', 'cameraman.jpg', 'dog.png']
    # file_path = './data/lena.png'
    file_path = './data/lena_noise.jpg'
    # file_path = './data/cameraman.jpg'
    # file_path = './data/dog.png'
    orig_img = Image.open(file_path).convert('RGB')
    orig_img = np.array(orig_img)

    rgb_rst = rgb(file_path)
    rgbxy_rst = rgbxy(file_path)
    filter_rst = filter_cluster(file_path)

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

    # plt.subplot(2, 2, 4)
    # plt.title('kmeans rgb sobel')
    # plt.imshow(rgb_sobel_rst)

    plt.show()