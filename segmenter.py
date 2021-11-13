import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering


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

    def find_clusters(self, n_clusters):
        """
        Method that clusters the data (of which the table data is already
          removed) using k-means clustering. Labels for each point, clusters
          of points, and the corresponding cluster centers are saved.
        """
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=0).fit(self.data_for_clustering)
        a_x = []
        a_y = []
        self.labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_
        self.clusters = [[] for i in range(n_clusters)]
        for i in range(len(self.data_for_clustering)):
            a_x.append(self.data_for_clustering[i][0])
            a_y.append(self.data_for_clustering[i][1])
            self.clusters[self.labels[i]].append([
                self.data_for_clustering[i][0], self.data_for_clustering[i][1]
            ])

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

        self.data_for_clustering = self.remove_table()

        self.find_clusters(n_clusters)

        self.translated_clusters = [[] for i in range(n_clusters)]
        self.new_depth_images = []
        self.new_rgb_images = []
        for i in range(n_clusters):
            self.translated_clusters[i] = self.translate_clusters(
                self.clusters[i], self.cluster_centers[i])
            new_depth_image, new_rgb_image = self.create_new_images(
                self.translated_clusters[i], self.clusters[i])
            self.new_depth_images.append(new_depth_image)
            self.new_rgb_images.append(new_rgb_image)

        segmentations = []
        for i in range(n_clusters):
            segmentations.append([
                self.cluster_centers[i], self.new_rgb_images[i],
                self.new_depth_images[i]
            ])

        return segmentations

    def remove_table(self):
        """
        Method for extracting data belonging to objects, and excluding data
          belonging to the table. That is, any point under the cutoff point
          (the median of the depth data) is used. One set of data is created,
          containing the depth and rgb values for each point in the original image.
          This data is returned to be used in clustering.
        """
        cutoff = np.median(self.depth_image)
        max_depth = np.max(self.depth_image)
        clustering_data = []
        for i in range(len(self.depth_image)):
            row = self.depth_image[i]
            for j in range(len(row)):
                if (self.depth_image[i][j] < cutoff):
                    # Depth is normalised based on the max depth in the data, rgb_image based on max possible value (255).
                    point = [
                        i, j, self.depth_image[i][j],
                        self.rgb_image[i][j][0] / 255,
                        self.rgb_image[i][j][1] / 255,
                        self.rgb_image[i][j][2] / 255
                    ]
                    clustering_data.append(point)

        return clustering_data

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