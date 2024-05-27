import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from CSV files
train_df = pd.read_csv('mnist/mnist_train.csv')
test_df = pd.read_csv('mnist/mnist_test.csv')

# Combine train and test for clustering purposes
full_df = pd.concat([train_df, test_df], axis=0)

# Separate features and labels
X = full_df.drop('label', axis=1).values
y = full_df['label'].values

# Normalize the pixel values
X = X / 255.0

# Apply PCA to reduce dimensionality
pca = PCA(n_components=50)  # Reduce to 50 dimensions
X_pca = pca.fit_transform(X)

# Apply k-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Map clusters to actual labels
# Create a DataFrame for easier manipulation
df = pd.DataFrame({'label': y, 'cluster': clusters})

# Find the most common label in each cluster
cluster_label_map = df.groupby('cluster')['label'].agg(lambda x: x.value_counts().index[0])

# Map the clusters to the most common label in that cluster
mapped_clusters = df['cluster'].map(cluster_label_map)

# Calculate the accuracy by comparing the mapped clusters to the true labels
accuracy = accuracy_score(y, mapped_clusters)
print(f'Clustering Accuracy: {accuracy}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y, mapped_clusters)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Visualize some of the clusters
def plot_clusters(data, clusters, labels, n_clusters=10, sample_size=100):
    plt.figure(figsize=(10, 10))
    for cluster in range(n_clusters):
        plt.subplot(1, n_clusters, cluster + 1)
        cluster_data = data[clusters == cluster]
        sampled_data = cluster_data[:sample_size]
        plt.imshow(sampled_data.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f'Cluster {cluster}')
    plt.show()

# Reduce to 2 dimensions for visualization purposes
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)

# Plot 2D clusters
plt.figure(figsize=(10, 7))
for cluster in range(10):
    cluster_data = X_pca_2d[clusters == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}', s=10)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.title('Clusters in 2D PCA Space')
plt.show()
