import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering


# Load in the depth data (essentially the depth image)
a = np.loadtxt("depth_data.txt", delimiter=",")

# Set the depth distance above which the mask will be applied,
#   such that the table parts of the data are removed. For now 
#   the median seems to be a good choice.
cutoff = np.median(a)

# Create a new list that only contains depth data below the 
#   cutoff. The result should contain only depth data for
#   the objects on the table.
a_correct = []
for i in range(len(a)):
    row = a[i]
    for j in range(len(row)):
        if (a[i][j] < cutoff):
            point = [i, j, a[i][j]]
            a_correct.append(point)

# Create a mask to remove the table, and plot the remaining points.
#   The colour shows the depth.
mask = np.zeros_like(a)
mask[a >= cutoff] =True
ax = sns.heatmap(a, mask=mask)

# Perform K-means clustering to find the centers of each object.
#   The found centers are plotted on top of the masked depth image.
kmeans = KMeans(n_clusters=5, random_state=0).fit(a_correct)
print(kmeans.cluster_centers_)
for row in kmeans.cluster_centers_:
    plt.scatter(x=row[1], y=row[0]) # x and y are switched because the created cluster centers are otherwise mirrored in line y=-x.


# Perform spectral clustering to find the five objects. 
# spectral = SpectralClustering(n_clusters=5, ).fit(a_correct)
# print(spectral.labels_)
# a_x = []
# a_y = []
# for point in a_correct:
#     a_x.append(point[0])
#     a_y.append(point[1])
# plt.scatter(a_y, a_x, c=spectral.labels_) # a_y and a_x are switched because the created clustered image is otherwise mirrored in line y=-x.


plt.show()

# Uncomment this to also create the original, unmasked depth image.
# ax = sns.heatmap(a)
# plt.show()
