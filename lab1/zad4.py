# V = (całkowity_dystans) / (całkowity czas = (2*d) / (d/v1 + d/v2) - gdy są zmiennymi losowymi niezależnymi
# V = (2d) / (d/v1 + d/(x*v1)) - gdy są zależnymi zmiennymi losowymi i v2 = x*v1

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Generate random data for v1 and v2
np.random.seed(42)
v1 = np.random.uniform(1, 100, 10000)  # Speeds between 1 and 100 km/h
v2 = np.random.uniform(1, 100, 10000)  # Speeds between 1 and 100 km/h

# Calculate the mean speed using the analytical formula
v_mean = (2 * v1 * v2) / (v1 + v2)

# Combine v1 and v2 into a single input array
# Each element of X is a pair [v1, v2]
X = np.vstack((v1, v2)).T

# Reshape v_mean to be a 2D array for compatibility with TensorFlow
Y = v_mean.reshape(-1, 1)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a simple neural network model
model = models.Sequential([
  layers.Dense(64, activation='relu', input_shape=(2,)),
  layers.Dense(64, activation='relu'),
  layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, Y_train, epochs=100, validation_split=0.1, verbose=2)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, Y_test)
print(f"Test Loss: {test_loss}")

from sklearn.metrics import mean_squared_error
from math import sqrt

# Generate more test data for predictions
np.random.seed(43)  # Ensure reproducibility
test_v1 = np.random.uniform(1, 100, 100)  # Generate 100 new random speeds
test_v2 = np.random.uniform(1, 100, 100)  # Generate 100 new random speeds
test_data = np.vstack((test_v1, test_v2)).T  # Combine into a single array

# Calculate actual mean speeds using the analytical formula for comparison
actual_means = (2 * test_v1 * test_v2) / (test_v1 + test_v2)

# Predict mean speeds using the neural network model
predicted_means = model.predict(test_data).flatten()  # Flatten to make it a 1D array

# Calculate RMSE between actual and predicted values
rmse = sqrt(mean_squared_error(actual_means, predicted_means))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Display some of the predicted vs actual values for comparison
for i in range(10):  # Display first 10 predictions
    print(f"v1={test_v1[i]:.2f}, v2={test_v2[i]:.2f} | Predicted Mean Speed: {predicted_means[i]:.2f} km/h, Actual Mean Speed: {actual_means[i]:.2f} km/h")
