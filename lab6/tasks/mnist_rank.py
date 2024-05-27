import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

# Load the dataset from CSV files
train_df = pd.read_csv('mnist/mnist_train.csv')
test_df = pd.read_csv('mnist/mnist_test.csv')

# Separate features and labels
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values
X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Normalize the pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Initialize classifiers
knn = KNeighborsClassifier(n_neighbors=3)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='linear', probability=True, random_state=42)

# Train classifiers
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Predict probabilities
knn_proba = knn.predict_proba(X_test)
rf_proba = rf.predict_proba(X_test)
svm_proba = svm.predict_proba(X_test)

# Rank predictions
knn_rank = np.apply_along_axis(rankdata, 1, knn_proba)
rf_rank = np.apply_along_axis(rankdata, 1, rf_proba)
svm_rank = np.apply_along_axis(rankdata, 1, svm_proba)

# Aggregate ranks
rank_sum = knn_rank + rf_rank + svm_rank
rank_agg_pred = np.argmax(rank_sum, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, rank_agg_pred)
print(f'Test Accuracy: {accuracy}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, rank_agg_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
print("Classification Report:")
print(classification_report(y_test, rank_agg_pred))

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
