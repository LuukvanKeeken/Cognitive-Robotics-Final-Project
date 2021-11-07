import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering



a = np.loadtxt("depth_data.txt", delimiter=",")
cutoff = np.median(a)
a_correct = []
for i in range(len(a)):
    row = a[i]
    for j in range(len(row)):
        if (a[i][j] < cutoff):
            point = [i, j, a[i][j]]
            a_correct.append(point)


# kmeans = KMeans(n_clusters=5, random_state=0).fit(a_correct)
# print(kmeans.cluster_centers_)

spectral = SpectralClustering(n_clusters=5, ).fit(a_correct)
print(spectral.labels_)
a_x = []
a_y = []
for point in a_correct:
    a_x.append(point[0])
    a_y.append(point[1])

plt.scatter(a_x, a_y, c=spectral.labels_)




# mask = np.zeros_like(a)
# mask[a >= cutoff] =True


# ax = sns.heatmap(a, mask=mask)
# for row in kmeans.cluster_centers_:
#     plt.scatter(x=row[0], y=row[1])

plt.show()

# ax = sns.heatmap(a)

# plt.show()
