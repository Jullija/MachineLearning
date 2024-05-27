import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tqdm import tqdm

# Function to load images
def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder), desc=f"Loading images from {folder}"):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Resize images to 224x224 for VGG16
            images.append(img)
    return images

# Load cat and dog images
cat_images = load_images_from_folder('path_to_cat_images_folder')
dog_images = load_images_from_folder('path_to_dog_images_folder')

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

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of VGG16
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_accuracies = []

for train_index, val_index in tqdm(kf.split(X_train), total=kf.get_n_splits(), desc="K-Fold Cross Validation"):
    X_train_kf, X_val_kf = X_train[train_index], X_train[val_index]
    y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]

    model.fit(X_train_kf, y_train_kf, epochs=10, batch_size=32, verbose=1)
    val_loss, val_acc = model.evaluate(X_val_kf, y_val_kf, verbose=0)
    val_accuracies.append(val_acc)
    print(f"Validation Accuracy: {val_acc}")

# Train final model
print("Training final model...")
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Test the model
print("Testing the model...")
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Test Accuracy: {accuracy}")
