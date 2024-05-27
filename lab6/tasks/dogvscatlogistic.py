import numpy as np
import os
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from tqdm import tqdm


# Function to load images
def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder), desc=f"Loading images from {folder}"):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize images to 64x64
            img = img.flatten()  # Flatten the image
            images.append(img)
    return images


# Load cat and dog images
cat_images = load_images_from_folder('test_set/cats')
dog_images = load_images_from_folder('test_set/dogs')

# Labeling images
cat_labels = [0] * len(cat_images)
dog_labels = [1] * len(dog_images)

# Combine and shuffle
images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)
images, labels = shuffle(images, labels, random_state=42)

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_accuracies = []

log_reg = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)

for train_index, val_index in tqdm(kf.split(X_train), total=kf.get_n_splits(), desc="K-Fold Cross Validation"):
    X_train_kf, X_val_kf = X_train[train_index], X_train[val_index]
    y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]

    log_reg.fit(X_train_kf, y_train_kf)
    val_predictions = log_reg.predict(X_val_kf)
    val_acc = accuracy_score(y_val_kf, val_predictions)
    val_accuracies.append(val_acc)
    print(f"Validation Accuracy: {val_acc}")

# Train final model
print("Training final model...")
log_reg.fit(X_train, y_train)

# Test the model
print("Testing the model...")
y_pred = log_reg.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Test Accuracy: {accuracy}")
