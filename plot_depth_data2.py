# This was created to get a view of the depth values present in an image.

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load in the depth data.
a = np.loadtxt("depth_data.txt", delimiter=",")

# Create a list of all depth values in the data and sort it.
all_values = []
for i in range(len(a)):
    row = a[i]
    for j in range(len(row)):
        all_values.append(a[i][j])

# Create a scatterplot of all depth data points, from smallest to largest.
all_values.sort()
ax = sns.scatterplot(x=range(0, len(all_values)), y=all_values)
print(all_values[6000])
print(all_values[3507])
print(np.median(all_values))
print(np.median(a))
plt.show()

# Create a histogram of the depth data
# ax = plt.hist(all_values)
# plt.show()