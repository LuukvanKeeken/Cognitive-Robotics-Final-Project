#import enum
#import random

import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#from sklearn.cluster import KMeans
#import sklearn.metrics as metrics
#import cv2
from skimage.segmentation import watershed
#from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.filters import rank

class Segmenter:
    def __init__(self):
        pass

    def create_new_images(self, translated_cluster, cluster):
        """
        Method that creates new depth and rgb images with a cluster of points
          at the center, and the rest removed. Images are initialised based on
          the shapes of the original depth and rgb images. Initially, each value
          is set to the depth or colour at (0, 0). Then, the method loops over
          each point in the cluster. At the new coordinates (the translated points),
          the depth or rgb values of the old coordinates (in the original images) are
          inserted.
        translated_cluster: points of the cluster when translated to the center.
        cluster: points of the cluster.
        """

        new_depth_image = np.full_like(self.depth_image,
                                       self.depth_image[0][0])
        new_rgb_image = np.full_like(self.rgb_image, self.rgb_image[0][0])

        for i in range(len(translated_cluster)):
            new_depth_image[translated_cluster[i][0]][
                translated_cluster[i][1]] = self.depth_image[cluster[i][0],
                                                             cluster[i][1]]
            new_rgb_image[translated_cluster[i][0]][
                translated_cluster[i][1]] = self.rgb_image[cluster[i][0],
                                                           cluster[i][1]]

        return new_depth_image, new_rgb_image


    def get_segmentations(self, rgb_image, depth_image, n_clusters):
        """
        Central method for coordinating the segmentation process. Firstly,
        the points beloning to objects (and not the table) are identified.
        This data is then clustered. For each cluster, a new depth image
        and a new rgb image is then created, with the cluster at the center
        and the rest of the object data removed. For each cluster, the cluster center
        is also returned. This is a 6-dimensional vector, containing the x and y
        pixel coordinates of the cluster center in the original picture, the depth
        of the cluster center, and the rgb values of the cluster center. Note, these
        depth and rgb values are not the values in the original images at that (x,y)
        location, but rather the calculted centers in depth and rgb space.

        n_clusters: the number of clusters that should be found in this image.
        rgb_image: original rgb image as taken by the robot, containing
          n_clusters objects.
        depth_image: the corresponding depth image.
        """
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        image = -np.array(depth_image*255).astype('uint8')

        #denoise image
        denoised = rank.median(image, disk(1))

        # find continuous region (low gradient -
        # where less than 10 for this image) --> markers
        # disk(5) is used here to get a more smooth image
        markers = rank.gradient(denoised, disk(3)) < 1
        markers = ndi.label(markers)[0]

        # local gradient (disk(2) is used to keep edges thin)
        gradient = rank.gradient(denoised, disk(1))

        # process the watershed
        labels = watershed(gradient, markers)

        if False:
            fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True, num=1)
            ax = axes.ravel()

            ax[0].imshow(image, cmap=plt.cm.gray)
            ax[0].set_title('Overlapping objects')
            ax[1].imshow(-100*gradient, cmap=plt.cm.gray)
            ax[1].set_title('Distances')
            ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
            ax[2].set_title('Separated objects')

            for a in ax:
                a.set_axis_off()

            fig.tight_layout()
            plt.show()

        clusters = [[] for i in range(np.max(labels)-1)]
        centers = [[] for i in range(np.max(labels)-1)]
        for y, row in enumerate(labels):
            for x, label in enumerate(row):
                label -= 2
                if label < 0:
                    continue
                clusters[label].append([x,y])
        for label, cluster in enumerate(clusters):
            centers[label] = np.mean(cluster, axis=0).astype(int)

        translated_clusters = [[] for i in range(np.max(labels)-1)]
        new_depth_images = []
        new_rgb_images = []
        for i in range(np.max(labels)-1):
            translated_clusters[i] = self.translate_clusters(
                clusters[i], centers[i])
            new_depth_image, new_rgb_image = self.create_new_images(
                translated_clusters[i], clusters[i])
            new_depth_images.append(new_depth_image)
            new_rgb_images.append(new_rgb_image)

        segmentations = []
        for i in range(np.max(labels)-1):
            segmentations.append([
                centers[i], new_rgb_images[i],
                new_depth_images[i]
            ])

        return segmentations

    def translate_clusters(self, cluster, cluster_center):
        """
        Method that calculates how much a cluster should be shifted
          for the cluster center to be at the center of the image (111, 111).
        cluster: set of point belonging to one cluster.
        cluster_center: the corresponding center of that cluster.
        """

        center_x = round(cluster_center[0])
        center_y = round(cluster_center[1])
        x_diff = 111 - center_x
        y_diff = 111 - center_y
        new_cluster = []

        for point in cluster:
            new_x = point[0] + x_diff
            new_y = point[1] + y_diff
            if new_x >= 0 and new_x < 224 and new_y >= 0 and new_y < 224:
                new_cluster.append([new_x, new_y])

        return new_cluster