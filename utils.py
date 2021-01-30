import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import util


def test_add_noise(file_path):
    """测试是否对图像添加不同的噪声
    """
    img = Image.open(file_path)
    img = np.array(img)
    noise_gs_img = util.random_noise(img,mode='gaussian')
    noise_salt_img = util.random_noise(img,mode='salt')
    noise_pepper_img = util.random_noise(img,mode='pepper')
    noise_sp_img = util.random_noise(img,mode='s&p')
    noise_speckle_img = util.random_noise(img,mode='speckle')

    plt.subplot(2,3,1), plt.title('original')
    plt.imshow(img)
    plt.subplot(2,3,2),plt.title('gaussian')
    plt.imshow(noise_gs_img)
    plt.subplot(2,3,3), plt.title('salt')
    plt.imshow(noise_salt_img)
    plt.subplot(2,3,4), plt.title('pepper')
    plt.imshow(noise_pepper_img)
    plt.subplot(2,3,5),plt.title('s&p')
    plt.imshow(noise_sp_img)
    plt.subplot(2,3,6), plt.title('speckle')
    plt.imshow(noise_speckle_img)
    plt.show()


def test_show_histogram(file_path):
    """展示灰度图像的直方图
    """
    img_gray = Image.open(file_path).convert('L')   
    img_gray = np.array(img_gray)
    hist, bins = np.histogram(img_gray.flatten(), bins=256, range=(0, 256))

    # draw histogram
    plt.hist(img_gray.flatten(), 256, [0, 256])
    plt.xlabel('gray value')
    plt.ylabel('count')
    plt.show()


if __name__ == '__main__':
    file_path = './data/coins.png'

    test_add_noise(file_path)
    # test_show_histogram(file_path)
    
