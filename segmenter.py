import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
import random



class Segmenter:
    def __init__(self):
        pass

    def create_new_depth_image(self, translated_cluster, cluster):
        new_image = np.full_like(self.depth_image, self.depth_image[0][0])
        for i in range(len(translated_cluster)):
            new_image[translated_cluster[i][0]][translated_cluster[i][1]] = self.depth_image[cluster[i][0], cluster[i][1]]

        new_image_plot = plt.figure()
        new_image_plot_ax = new_image_plot.add_subplot()
        new_image_plot_ax = sns.heatmap(new_image)

        return new_image
        

    def create_new_rgb_image(self, translated_cluster, cluster):
        new_image = np.full_like(self.rgb_image, self.rgb_image[0][0])
        new_image_test = np.full_like(self.depth_image, self.rgb_image[0][0][0])
        for i in range(len(translated_cluster)):
            new_image[translated_cluster[i][0]][translated_cluster[i][1]] = self.rgb_image[cluster[i][0], cluster[i][1]]
            new_image_test[translated_cluster[i][0]][translated_cluster[i][1]] = self.rgb_image[cluster[i][0]][cluster[i][1]][0]
        
        new_image_plot = plt.figure()
        new_image_plot_ax = new_image_plot.add_subplot()
        new_image_plot_ax = sns.heatmap(new_image_test)

        rgb_img_plot = plt.figure()
        rgb_ax = rgb_img_plot.add_subplot()

        
        for i in range(81, 141):
            print(i)
            for j in range(81, 141):
                rgb_ax.scatter(j, i, c=[[new_image[i][j][0]/255, new_image[i][j][1]/255, new_image[i][j][2]/255]])

        



    def find_clusters(self):
        kmeans = KMeans(n_clusters=5, random_state=0).fit(self.data_for_clustering)
        a_x = []
        a_y = []
        self.labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_
        self.clusters = [[] for i in range(5)]
        for i in range(len(self.data_for_clustering)):
            a_x.append(self.data_for_clustering[i][0])
            a_y.append(self.data_for_clustering[i][1])
            self.clusters[self.labels[i]].append([self.data_for_clustering[i][0], self.data_for_clustering[i][1]])
        

    def get_segmentations(self, rgb_image, depth_image):
        self.rgb_image = rgb_image
        self.depth_image = depth_image

        depth_fig = plt.figure()
        depth_figax = depth_fig.add_subplot()
        depth_figax = sns.heatmap(depth_image)

        self.data_for_clustering = self.remove_table()

        self.find_clusters()

        self.translated_clusters = [[] for i in range(5)]
        for i in range(5):
            self.translated_clusters[i] = self.translate_clusters(self.clusters[i], self.cluster_centers[i])

        self.new_depth_images = []
        self.new_rgb_images = []
        for i in range(1):
            self.new_depth_images.append(self.create_new_depth_image(self.translated_clusters[i], self.clusters[i]))
            self.new_rgb_images.append(self.create_new_rgb_image(self.translated_clusters[i], self.clusters[i]))
        plt.show()
        exit(0)
        segmentations = []
        for i in range(5):
            segmentations.append[self.cluster_centers[i], self.new_depth_images[i],] 


    
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
                    point = [i, j, self.depth_image[i][j], self.rgb_image[i][j][0]/255, self.rgb_image[i][j][1]/255, self.rgb_image[i][j][2]/255]
                    clustering_data.append(point)

        return clustering_data

    
    def translate_clusters(self, cluster, cluster_center):
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
                



