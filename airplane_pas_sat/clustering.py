import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv('airplane_pas_sat/train.csv')
test = pd.read_csv('airplane_pas_sat/test.csv')

# Encode categorical features
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
encoders = {col: LabelEncoder().fit(train[col]) for col in categorical_cols}
df = pd.concat([train, test], ignore_index=True)
for col in categorical_cols:
    df[col] = encoders[col].transform(df[col])

# Prepare features and labels
df_sampled = df.sample(frac=0.3, random_state=42)  # use 30% of data
X = df_sampled.drop(columns=["Unnamed: 0", "id", "satisfaction"]).fillna(0)
y = df_sampled['satisfaction']
X_scaled = StandardScaler().fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

agg = AgglomerativeClustering(n_clusters=2, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)

# Assign cluster majority labels
def assign_majority_labels(cluster_labels, true_labels):
    cluster_to_label = {}
    for cluster in np.unique(cluster_labels):
        mask = (cluster_labels == cluster)
        majority = true_labels[mask].mode()[0]
        cluster_to_label[cluster] = majority
    return np.array([cluster_to_label[c] for c in cluster_labels])

# Get predicted labels
kmeans_pred = assign_majority_labels(kmeans_labels, y)
agg_pred = assign_majority_labels(agg_labels, y)

# Accuracy
kmeans_acc = accuracy_score(y, kmeans_pred)
agg_acc = accuracy_score(y, agg_pred)

print(f'KMeans Accuracy: {kmeans_acc:.4f}')
print(f'Agglomerative Clustering Accuracy: {agg_acc:.4f}')
