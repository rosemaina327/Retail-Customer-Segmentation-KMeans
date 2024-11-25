import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# One-hot encode Gender
data_encoded = pd.get_dummies(data, columns=['Gender'], drop_first=True)

# Selecting features for clustering
features = data_encoded[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using Silhouette Scores
silhouette_scores = []
wcss = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, clusters)
    silhouette_scores.append(score)
    wcss.append(kmeans.inertia_)
    print(f"k={k}, Silhouette Score={score:.4f}")

# Plot Silhouette Scores and WCSS
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Scores for Different k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

plt.tight_layout()
plt.show()

# Based on the above, let's assume the optimal k is 3
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add the cluster labels to the dataset
data['Cluster'] = clusters

# Analyze each cluster
for i in range(optimal_clusters):
    print(f"\nCluster {i} Characteristics:")
    print(data[data['Cluster'] == i][['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe())

# Visualize clusters using scatter plot
sns.scatterplot(
    x=data['Annual Income (k$)'], 
    y=data['Spending Score (1-100)'], 
    hue=data['Cluster'], 
    palette='Set2'
)
plt.title('Clusters based on Annual Income and Spending Score')
plt.show()

# PCA Visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

plt.figure(figsize=(8, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, cmap='viridis', s=50)
plt.title('PCA Visualization of Clusters')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster')
plt.show()
