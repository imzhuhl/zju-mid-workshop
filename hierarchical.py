import time as time
import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering


if __name__ == '__main__':
    # load image
    file_path = './data/cameraman.jpg'
    img_gray = Image.open(file_path).convert('L')
    img_gray = np.array(img_gray)
    X = np.reshape(img_gray, (-1, 1))

    # Define the structure A of the data. Pixels connected to their neighbors.
    connectivity = grid_to_graph(*img_gray.shape)

    print("Compute structured hierarchical clustering...")
    st = time.time()
    n_clusters = 2  # number of regions
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity).fit(X)
    label = np.reshape(ward.labels_, img_gray.shape)
    print("Elapsed time: ", time.time() - st)
    print("Number of pixels: ", label.size)
    print("Number of clusters: ", np.unique(label).size)

    # Plot the results on an image
    plt.imshow(label)
    plt.show()

