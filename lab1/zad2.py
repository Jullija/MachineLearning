import numpy as np
from sklearn.datasets import make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt


# Function to train a logistic regression model with polynomial features of varying degrees
def train_polynomial_logistic_regression(X, y, degree):
    if degree < 2:
        raise ValueError("Degree should be at least 2 for polynomial logistic regression.")
    pipeline = Pipeline([
        ("polynomial_features", PolynomialFeatures(degree)),
        ("logistic_regression", LogisticRegression())
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    return pipeline


# Function to plot the decision boundary of a model
def plot_decision_boundary(model, X, y, degree, ax):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(f"Degree {degree} polynomial logistic regression")


# Generate circular data
X_circles, y_circles = make_circles(n_samples=100, noise=0.1, factor=0.2, random_state=0)

# Generate linearly separable data
X_linear, y_linear = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1,
                                         n_redundant=0, n_informative=2, random_state=42)

# Plot decision boundaries for models with polynomial degrees 1 (linear), 2, and 3
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Plot for circular data with degrees 1, 2, and 3
for i, degree in enumerate([1, 2, 3]):
    if degree == 1:
        model = LogisticRegression()
        model.fit(X_circles, y_circles)
    else:
        model = train_polynomial_logistic_regression(X_circles, y_circles, degree)
    ax = axs[0, i]
    plot_decision_boundary(model, X_circles, y_circles, degree, ax)

# Plot for linearly separable data with degrees 1, 2, and 3
for i, degree in enumerate([1, 2, 3]):
    if degree == 1:
        model = LogisticRegression()
        model.fit(X_linear, y_linear)
    else:
        model = train_polynomial_logistic_regression(X_linear, y_linear, degree)
    ax = axs[1, i]
    plot_decision_boundary(model, X_linear, y_linear, degree, ax)

plt.tight_layout()
plt.show()
