import numpy as np
import os
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tqdm import tqdm

# Function to load images
def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder), desc=f"Loading images from {folder}"):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize images to 64x64
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

# Normalize images
images = images / 255.0

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_accuracies = []

for train_index, val_index in tqdm(kf.split(X_train), total=kf.get_n_splits(), desc="K-Fold Cross Validation"):
    X_train_kf, X_val_kf = X_train[train_index], X_train[val_index]
    y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]

    model = create_model()
    model.fit(X_train_kf, y_train_kf, epochs=10, batch_size=32, verbose=1)
    val_loss, val_acc = model.evaluate(X_val_kf, y_val_kf, verbose=0)
    val_accuracies.append(val_acc)
    print(f"Validation Accuracy: {val_acc}")

# Train final model
print("Training final model...")
final_model = create_model()
final_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Test the model
print("Testing the model...")
y_pred = (final_model.predict(X_test) > 0.5).astype("int32")

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Test Accuracy: {accuracy}")

