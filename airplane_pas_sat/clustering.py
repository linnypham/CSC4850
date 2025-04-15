from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

num_features = ['Age', 'Flight Distance', 'Inflight wifi service', 
                'Departure/Arrival time convenient', 'Ease of Online booking', 
                'Gate location', 'Food and drink', 'Online boarding', 
                'Seat comfort', 'Inflight entertainment', 'On-board service', 
                'Leg room service', 'Baggage handling', 'Checkin service', 
                'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 
                'Arrival Delay in Minutes']
cat_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_processed)
from sklearn.cluster import AgglomerativeClustering
agglo = AgglomerativeClustering(n_clusters=2)
clusters_agglo = agglo.fit_predict(X_processed)
import pandas as pd
from scipy.stats import mode

def assign_labels(clusters, y_true):
    df = pd.DataFrame({'cluster': clusters, 'true_label': y_true})
    majority_labels = df.groupby('cluster')['true_label'].agg(lambda x: mode(x)[0])
    df['predicted_label'] = df['cluster'].map(majority_labels)
    return df['predicted_label']
# Load and prepare data
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
df = df.drop(columns=['id'])  # Remove irrelevant column
X = df.drop(columns=['satisfaction'])
y = df['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1})

# Preprocess features
X_processed = preprocessor.fit_transform(X)

# Apply clustering
kmeans_clusters = KMeans(n_clusters=2, random_state=42).fit_predict(X_processed)
agglo_clusters = AgglomerativeClustering(n_clusters=2).fit_predict(X_processed)

# Evaluate accuracy
from sklearn.metrics import accuracy_score

y_pred_kmeans = assign_labels(kmeans_clusters, y)
y_pred_agglo = assign_labels(agglo_clusters, y)

print(f"K-Means Accuracy: {accuracy_score(y, y_pred_kmeans):.3f}")
print(f"Agglomerative Accuracy: {accuracy_score(y, y_pred_agglo):.3f}")
