import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv('airplane_pas_sat/train.csv', index_col=0)
test = pd.read_csv('airplane_pas_sat/test.csv', index_col=0)
data = pd.concat([train, test])

# Handle missing values for categorical columns
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
imputer_cat = SimpleImputer(strategy='most_frequent')  # For categorical columns
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

# Handle missing values for numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputer_num = SimpleImputer(strategy='mean')  # For numerical columns
data[numerical_cols] = imputer_num.fit_transform(data[numerical_cols])

# Convert 'satisfaction' to binary labels
data['satisfaction'] = data['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

# One-hot encode categorical columns
X = pd.get_dummies(data.drop(['id', 'satisfaction'], axis=1), columns=categorical_cols)
y = data['satisfaction']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Apply Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=2)
agg_labels = agg.fit_predict(X_scaled)

# Define function to evaluate clusters
def evaluate_clusters(true_labels, cluster_labels):
    cluster_df = pd.DataFrame({'cluster': cluster_labels, 'true_label': true_labels})
    majority_label = cluster_df.groupby('cluster')['true_label'].agg(lambda x: x.mode()[0])
    cluster_df['predicted'] = cluster_df['cluster'].map(majority_label)
    return accuracy_score(cluster_df['true_label'], cluster_df['predicted'])

# Evaluate models
kmeans_accuracy = evaluate_clusters(y, kmeans_labels)
agg_accuracy = evaluate_clusters(y, agg_labels)

# Full workflow
print(f"K-Means Accuracy: {kmeans_accuracy:.3f}")
print(f"Agglomerative Accuracy: {agg_accuracy:.3f}")
ccc