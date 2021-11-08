import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
import random


# Load in the depth data (essentially the depth image)
a = np.loadtxt("depth_data.txt", delimiter=",")

r = np.loadtxt("r_colour_data.txt", delimiter=",")
g = np.loadtxt("g_colour_data.txt", delimiter=",")
b = np.loadtxt("b_colour_data.txt", delimiter=",")

rgb = np.concatenate((r, g, b)).reshape((3, 224, 224)).transpose((1, 2, 0))



# Set the depth distance above which the mask will be applied,
#   such that the table parts of the data are removed. For now 
#   the median seems to be a good choice. 
cutoff = np.median(a)
max_depth = np.max(a)
a_correct = []
for i in range(len(a)):
    row = a[i]
    for j in range(len(row)):
        if (a[i][j] < cutoff):
            # Depth is normalised based on the max depth in the data, rgb based on max possible value (255).
            point = [i, j, a[i][j]/max_depth, rgb[i][j][0]/255, rgb[i][j][1]/255, rgb[i][j][2]/255]
            a_correct.append(point)


# Create a mask to remove the table, and plot the remaining points.
#   The colour shows the depth.
mask = np.zeros_like(a)
mask[a >= cutoff] =True
ax = sns.heatmap(a, mask=mask)


# Perform K-means clustering to find the centers of each object.
#   The found centers are plotted on top of the masked depth image.
#   Uncomment this to use this method.
# kmeans = KMeans(n_clusters=5, random_state=0).fit(a_correct)
# print(kmeans.cluster_centers_)
# for row in kmeans.cluster_centers_:
#     plt.scatter(x=row[1], y=row[0]) # x and y are switched because the created cluster centers are otherwise mirrored in line y=-x.


# Perform spectral clustering to find the five objects. Comment/uncomment this
# whole block to disable/enable spectral clustering.
spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', n_neighbors=20).fit(a_correct)
# Plot the data points, which are coloured based on which 
#   cluster they belong to. Only half of the points are
#   plotted (randomly), because otherwise it takes too 
#   much time.
a_x = []
a_y = []
spectral_lables  = []
for i in range(len(a_correct)):
    a_x.append(a_correct[i][0])
    a_y.append(a_correct[i][1])
    spectral_lables.append(spectral.labels_[i])
plt.scatter(a_y, a_x, c=spectral_lables) # a_y and a_x are switched because the created clustered image is otherwise mirrored in line y=-x.


plt.show()






