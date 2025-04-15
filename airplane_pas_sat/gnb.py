from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Load datasets
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

# Identify categorical columns
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

# Create a dictionary to store encoders for each column
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])  # Might raise error if unseen labels exist
    encoders[col] = le

# Split features and labels
X_train = train.drop(columns=["Unnamed: 0", "id", "satisfaction"]).fillna(0)
y_train = train['satisfaction']

X_test = test.drop(columns=["Unnamed: 0", "id", "satisfaction"]).fillna(0)
y_test = test['satisfaction']

# Train GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

# Compute log probability differences
log_prob_diff = np.abs(
    model.theta_[1] / np.sqrt(model.var_[1]) - 
    model.theta_[0] / np.sqrt(model.var_[0])
)

# Get top discriminative feature names
top_words_indices = np.argsort(log_prob_diff)[::-1][:10]
words = X_train.columns  # Now defined properly
top_words = [words[i] for i in top_words_indices]

print("Top 10 discriminative features:")
for word in top_words:
    print(word)
train_accuracy = model.score(X_train, y_train) * 100
test_accuracy = model.score(X_test, y_test) * 100

print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")