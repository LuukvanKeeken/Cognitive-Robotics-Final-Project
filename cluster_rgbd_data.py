import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
import random


def plot_clusters(a_correct, labels):
    a_x = []
    a_y = []
    clusters = [[], [], [], [], []]
    for i in range(len(a_correct)):
        a_x.append(a_correct[i][0])
        a_y.append(a_correct[i][1])
        # spectral_lables.append(spectral.labels_[i])
        clusters[labels[i]].append([a_correct[i][0], a_correct[i][1]])
    plt.scatter(a_y, a_x, c=labels) # a_y and a_x are switched because the created clustered image is otherwise mirrored in line y=-x.

    return clusters


def k_means_clustering(a_correct):
    kmeans = KMeans(n_clusters=5, random_state=0).fit(a_correct)


def spectral_clustering(a_correct):
    spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', n_neighbors=20).fit(a_correct)
    return spectral.labels_




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
            point = [i, j, a[i][j]*1000, rgb[i][j][0]/255, rgb[i][j][1]/255, rgb[i][j][2]/255]
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
labels = k_means_clustering(a_correct)
# a_x = []
# a_y = []
# lables = []
# for i in range(len(a_correct)):
#     a_x.append(a_correct[i][0])
#     a_y.append(a_correct[i][1])
#     lables.append(kmeans.labels_[i])
# plt.scatter(a_y, a_x, c=lables)    
# for row in kmeans.cluster_centers_:
#     plt.scatter(x=row[1], y=row[0], c="orange") # x and y are switched because the created cluster centers are otherwise mirrored in line y=-x.
# plt.show()


# Perform spectral clustering to find the five objects. Comment/uncomment this
# whole block to disable/enable spectral clustering.
# spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', n_neighbors=20).fit(a_correct)
# Plot the data points, which are coloured based on which 
#   cluster they belong to. Only half of the points are
#   plotted (randomly), because otherwise it takes too 
#   much time.

labels  = spectral_clustering(a_correct)
clusters = plot_clusters(a_correct, labels)


# fig = plt.figure()
# ax = fig.add_subplot()
# ax = sns.heatmap(a)
# plt.show()

def bounding_box(coords):
    min_x = 100000 # start with something much higher than expected min
    min_y = 100000
    max_x = -100000 # start with something much lower than expected max
    max_y = -100000

    for item in coords:
        if item[0] < min_x:
            min_x = item[0]

        if item[0] > max_x:
            max_x = item[0]

        if item[1] < min_y:
            min_y = item[1]

        if item[1] > max_y:
            max_y = item[1]

    return [(min_x,min_y),(max_x,min_y),(max_x,max_y),(min_x,max_y)]



# mask = np.ones_like(a)

# for point in clusters[0]:
#     mask[point[0]][point[1]] = False

bot_left, bot_right, top_right, top_left = bounding_box(clusters[0])

# fig2 = plt.figure()
# ax2 = fig2.add_subplot()
# ax2 = sns.heatmap(a, mask=mask)
ax.scatter(x=bot_left[1], y=bot_left[0], c = "red")
ax.scatter(x=bot_right[1], y=bot_right[0], c= "red")
ax.scatter(x=top_right[1], y=top_right[0], c="red")
ax.scatter(x=top_left[1], y=top_left[0], c="red")
# ax2.plot(bot_left, bot_right)
# print(bot_left, bot_right)


# mask = np.ones_like(a)

# for point in clusters[1]:
#     mask[point[0]][point[1]] = False

bot_left, bot_right, top_right, top_left = bounding_box(clusters[1])
# fig3 = plt.figure()
# ax3 = fig3.add_subplot()
# ax3 = sns.heatmap(a, mask=mask)

ax.scatter(x=bot_left[1], y=bot_left[0], c="blue")
ax.scatter(x=bot_right[1], y=bot_right[0], c="blue")
ax.scatter(x=top_right[1], y=top_right[0], c="blue")
ax.scatter(x=top_left[1], y=top_left[0], c="blue")


# mask = np.ones_like(a)

# for point in clusters[2]:
#     mask[point[0]][point[1]] = False
bot_left, bot_right, top_right, top_left = bounding_box(clusters[2])
# fig4 = plt.figure()
# ax4 = fig4.add_subplot()
# ax4 = sns.heatmap(a, mask=mask)
ax.scatter(x=bot_left[1], y=bot_left[0], c="yellow")
ax.scatter(x=bot_right[1], y=bot_right[0], c="yellow")
ax.scatter(x=top_right[1], y=top_right[0], c="yellow")
ax.scatter(x=top_left[1], y=top_left[0], c="yellow")



# mask = np.ones_like(a)

# for point in clusters[3]:
#     mask[point[0]][point[1]] = False
bot_left, bot_right, top_right, top_left = bounding_box(clusters[3])
# fig5 = plt.figure()
# ax5 = fig5.add_subplot()
# ax5 = sns.heatmap(a, mask=mask)
ax.scatter(x=bot_left[1], y=bot_left[0], c="green")
ax.scatter(x=bot_right[1], y=bot_right[0], c="green")
ax.scatter(x=top_right[1], y=top_right[0], c="green")
ax.scatter(x=top_left[1], y=top_left[0], c="green")



# mask = np.ones_like(a)

# for point in clusters[4]:
#     mask[point[0]][point[1]] = False
bot_left, bot_right, top_right, top_left = bounding_box(clusters[4])
# fig6 = plt.figure()
# ax6 = fig6.add_subplot()
# ax6 = sns.heatmap(a, mask=mask)

ax.scatter(x=bot_left[1], y=bot_left[0], c="orange")
ax.scatter(x=bot_right[1], y=bot_right[0], c="orange")
ax.scatter(x=top_right[1], y=top_right[0], c="orange")
ax.scatter(x=top_left[1], y=top_left[0], c="orange")


plt.show()






