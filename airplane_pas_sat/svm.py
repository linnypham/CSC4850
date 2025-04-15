import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Encode categorical columns
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    encoders[col] = le

# Select 2 top discriminative features for plotting
X_all = train.drop(columns=["Unnamed: 0", "id", "satisfaction"]).fillna(0)
y_all = train['satisfaction']

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_all, y_all)
log_prob_diff = np.abs(nb.theta_[1] / np.sqrt(nb.var_[1]) - nb.theta_[0] / np.sqrt(nb.var_[0]))
top_two_idx = np.argsort(log_prob_diff)[::-1][:2]
top_two_features = X_all.columns[top_two_idx]

# Use only 2 features for SVM and plot
X_train = train[top_two_features]
y_train = train['satisfaction']
X_test = test[top_two_features]
y_test = test['satisfaction']

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to plot decision boundary
def plot_decision_boundary(clf, X, y, title):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X, response_method="predict",
        cmap=plt.cm.coolwarm, alpha=0.8
    )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    disp.ax_.set_title(title)
    plt.xlabel(top_two_features[0])
    plt.ylabel(top_two_features[1])

# Train and plot for both kernels
kernels = ['linear', 'poly']
for kernel in kernels:
    clf = SVC(kernel=kernel, degree=3) 
    clf.fit(X_train_scaled, y_train)
    train_acc = clf.score(X_train_scaled, y_train) * 100
    test_acc = clf.score(X_test_scaled, y_test) * 100
    print(f"\nSVM ({kernel} kernel):")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Testing Accuracy: {test_acc:.2f}%")
    plt.figure()
    plot_decision_boundary(clf, X_train_scaled, y_train, f"SVM with {kernel} kernel")
    plt.savefig(f"SVM ({kernel} kernel).png")
print('done')