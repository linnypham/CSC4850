import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load datasets
test = pd.read_csv('airplane_pas_sat/test.csv')
train = pd.read_csv('airplane_pas_sat/train.csv')

# Identify categorical columns
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

# Encode categorical columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])  # Be careful: may fail if unseen labels
    encoders[col] = le

# Split features and labels
X_train = train.drop(columns=["Unnamed: 0", "id", "satisfaction"]).fillna(0)
y_train = train['satisfaction']

X_test = test.drop(columns=["Unnamed: 0", "id", "satisfaction"]).fillna(0)
y_test = test['satisfaction']

# Combine for clustering (unsupervised doesn't use train/test)
X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to assign majority label to each cluster
def majority_labeling(cluster_labels, true_labels, n_clusters):
    assigned_labels = np.zeros_like(cluster_labels)
    for i in range(n_clusters):
        mask = (cluster_labels == i)
        if np.sum(mask) == 0: continue
        assigned_labels[mask] = mode(true_labels[mask], keepdims=True)[0]
    return assigned_labels

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_majority_labels = majority_labeling(kmeans_labels, y.values, 2)
kmeans_acc = accuracy_score(y, kmeans_majority_labels)

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=2)
agglo_labels = agglo.fit_predict(X_scaled)
agglo_majority_labels = majority_labeling(agglo_labels, y.values, 2)
agglo_acc = accuracy_score(y, agglo_majority_labels)

# Print accuracies
print(f"KMeans Accuracy: {kmeans_acc:.4f}")
print(f"Agglomerative Clustering Accuracy: {agglo_acc:.4f}")

# Optional: PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title('KMeans Clustering')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)

plt.subplot(1, 2, 2)
plt.title('Agglomerative Clustering')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agglo_labels, cmap='plasma', alpha=0.6)

plt.tight_layout()
plt.show()
