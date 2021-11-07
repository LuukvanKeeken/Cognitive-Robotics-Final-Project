import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt("depth_data.txt", delimiter=",")
all_values = []

for i in range(len(a)):
    row = a[i]
    for j in range(len(row)):
        all_values.append(a[i][j])

all_values.sort()

ax = sns.scatterplot(x=range(0, 4000), y=all_values[0:4000])
print(all_values[6000])
print(all_values[3507])
print(np.median(all_values))
print(np.median(a))
plt.show()