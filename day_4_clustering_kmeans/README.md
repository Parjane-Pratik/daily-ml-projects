# Day 4: Clustering & Unsupervised Learning

## 📚 Learning Objectives

By the end of this day, you should understand:

- ✅ What is unsupervised learning?
- ✅ How clustering works
- ✅ Implement K-Means clustering
- ✅ Determine optimal number of clusters
- ✅ Evaluate clustering quality
- ✅ Explore other clustering algorithms

## 🎯 Key Concepts

### 1. Clustering Overview
Unsupervised learning to group similar data points together.

**Common Algorithms:**
- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models (GMM)

### 2. K-Means Algorithm

**Steps:**
1. Choose K (number of clusters)
2. Initialize K centroids randomly
3. Assign each point to nearest centroid
4. Update centroids based on assigned points
5. Repeat until convergence

**Pros:** Fast, scalable, easy to understand
**Cons:** Requires K specification, sensitive to initialization

### 3. Determining Optimal K

- **Elbow Method:** Plot inertia vs K, look for elbow
- **Silhouette Score:** Measure cluster quality (-1 to 1)
- **Davies-Bouldin Index:** Lower is better
- **Calinski-Harabasz Index:** Higher is better

### 4. Evaluation Metrics

- **Inertia:** Sum of squared distances to nearest centroid
- **Silhouette Score:** Measures how similar points are to own cluster vs others
- **Davies-Bouldin Index:** Average cluster separation
- **Homogeneity:** All samples in cluster belong to same class
- **Completeness:** All samples of class assigned to same cluster

## 📊 Dataset

**Mall Customer Segmentation** or **Iris Dataset**.

```bash
kaggle datasets download -d akram24/customers -p ./day_4_clustering_kmeans/data/ --unzip
```

## 🔧 Implementation

### K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and prepare data
X = df[['Annual Income', 'Spending Score']].values

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Determine optimal K using Elbow Method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 4. Visualize elbow
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')
plt.show()

# 5. Train with optimal K
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# 6. Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title(f'K-Means Clustering (K={optimal_k})')
plt.legend()
plt.show()
```

### Other Clustering Algorithms

```python
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
h_clusters = hierarchical.fit_predict(X_scaled)

# DBSCAN - Density-based
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_clusters = dbscan.fit_predict(X_scaled)

# Gaussian Mixture Models
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_clusters = gmm.fit_predict(X_scaled)
```

## 📝 Exercise

1. Load a clustering dataset
2. Prepare data:
   - Handle missing values
   - Scale features
3. Determine optimal number of clusters:
   - Use Elbow Method
   - Use Silhouette Score
   - Compare results
4. Train K-Means with optimal K:
   - Fit model
   - Get cluster assignments
5. Evaluate clustering:
   - Calculate Silhouette Score
   - Analyze cluster characteristics
6. Visualize:
   - Elbow plot
   - Silhouette score plot
   - Cluster scatter plots
   - Cluster centers
7. Try alternative algorithms:
   - Hierarchical Clustering
   - DBSCAN
   - Compare results

## 💡 Tips

1. **Always scale data:** Clustering is distance-based
2. **Elbow may not be clear:** Use Silhouette Score as backup
3. **Try multiple algorithms:** Different data benefits from different methods
4. **Domain knowledge:** Use business understanding to validate clusters
5. **Visualize results:** 2D/3D plots help validate clustering

## 🔗 Resources

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [K-Means Deep Dive](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Cluster Evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)

---

**Next:** Day 5 introduces neural networks and deep learning!