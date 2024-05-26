import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd


np.random.seed(42)

X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=10)


df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['Cluster'] = y  


df.to_csv('cluster_data.csv', index=False)


plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()