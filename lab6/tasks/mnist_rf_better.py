import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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

# Define base learners
base_learners = [
    ('rf1', RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1)),
    ('rf2', RandomForestClassifier(n_estimators=200, random_state=42,n_jobs=-1)),
    ('et1', ExtraTreesClassifier(n_estimators=100, random_state=42,n_jobs=-1)),
    ('et2', ExtraTreesClassifier(n_estimators=200, random_state=42,n_jobs=-1))
]

# Define meta learner
meta_learner = LogisticRegression()

# Create the Stacking classifier
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5, n_jobs=-1)

# Train the Stacking classifier
stacking_clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = stacking_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
