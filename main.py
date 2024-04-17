import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Display the initial rows of the dataset
print("Initial rows of the dataset:")
print(data.head())

# Select relevant features for clustering
X = data.iloc[:, 2:].values  # Selecting Age, Annual Income, and Spending Score

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method to visualize the optimal number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')  # Within-cluster sum of squares
plt.xticks(np.arange(1, 11, 1))
plt.grid(True)
plt.show()

# Based on the Elbow method, let's choose 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)

# Fit K-means to the standardized data
kmeans.fit(X_scaled)

# Assign clusters to the data points
clusters = kmeans.predict(X_scaled)

# Add the cluster labels to the original dataset
data['Cluster'] = clusters

# Visualize the clusters using pairplot
plt.figure(figsize=(12, 6))
sns.pairplot(data=data, hue='Cluster', palette='viridis', diag_kind='hist', diag_kws={'alpha':0.5})
plt.suptitle('Pairplot of Clusters', y=1.02)
plt.show()

# Visualize the clusters in 3D space
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = X_scaled[:, 0]
y = X_scaled[:, 1]
z = X_scaled[:, 2]

scatter = ax.scatter(x, y, z, c=clusters, cmap='viridis', marker='o')
ax.set_xlabel('Standardized Age')
ax.set_ylabel('Standardized Annual Income')
ax.set_zlabel('Standardized Spending Score')
plt.title('Clusters in 3D Space')
plt.legend(*scatter.legend_elements(), title='Clusters')
plt.show()

# Print the cluster centers
print("Cluster Centers:")
print(scaler.inverse_transform(kmeans.cluster_centers_))

# Save the updated dataset with cluster labels
data.to_csv("customer_data_with_clusters.csv", index=False)
