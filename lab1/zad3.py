import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Set random seed for reproducibility
np.random.seed(42)

# Define more interesting test functions
def cubic_function(X):
    return -0.4 * X ** 3 + 0.5 * X ** 2 + X

def sine_function(X):
    return np.sin(X)

def linear_function(X):
    return 2 * X + 3

def quadratic_function(X):
    return 0.5 * X ** 2 - 3 * X + 5

def sinusoidal_function(X):
    return 5 * np.sin(2 * np.pi * X / 10) + 3 * np.cos(2 * np.pi * X / 8)

# Generate data for the new test functions
X = np.linspace(-10, 10, 500)
y_cubic = cubic_function(X) + np.random.randn(*X.shape) * 20
y_sine = sine_function(X) + np.random.randn(*X.shape) * 0.5
y_linear = linear_function(X) + np.random.randn(*X.shape) * 10
y_quadratic = quadratic_function(X) + np.random.randn(*X.shape) * 5
y_sinusoidal = sinusoidal_function(X) + np.random.randn(*X.shape) * 5

datasets = [(X, y_linear), (X, y_quadratic), (X, y_cubic), (X, y_sine), (X, y_sinusoidal)]

# Define function to train models and return predictions along with MSE
def calculate_additional_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def train_and_evaluate(X, y, degrees, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    predictions = {}
    metrics = {degree: {'MSE': [], 'RMSE': [], 'MAE': [], 'R2': []} for degree in degrees}
    models = {}

    for degree in degrees:
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if degree == 'sinusoidal':
                model = LinearRegression()
                X_train_2 = np.column_stack((np.sin(X_train), np.cos(X_train)))
                X_test_2 = np.column_stack((np.sin(X_test), np.cos(X_test)))
                model.fit(X_train_2, y_train)
                y_test_pred = model.predict(X_test_2)
            else:
                poly_features = PolynomialFeatures(degree=degree)
                X_train_poly = poly_features.fit_transform(X_train[:, np.newaxis])
                X_test_poly = poly_features.transform(X_test[:, np.newaxis])

                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                y_test_pred = model.predict(X_test_poly)

            mse, rmse, mae, r2 = calculate_additional_metrics(y_test, y_test_pred)

            metrics[degree]['MSE'].append(mse)
            metrics[degree]['RMSE'].append(rmse)
            metrics[degree]['MAE'].append(mae)
            metrics[degree]['R2'].append(r2)

        # After the k-fold loop, calculate the mean of each metric for the current degree
        for metric in metrics[degree]:
            metrics[degree][metric] = np.mean(metrics[degree][metric])

        # Generate predictions for the entire dataset for plotting
        if degree == 'sinusoidal':
            X_2 = np.column_stack((np.sin(X), np.cos(X)))
            predictions[degree] = model.predict(X_2)
        else:
            X_poly = poly_features.fit_transform(X[:, np.newaxis])
            predictions[degree] = model.predict(X_poly)
        models[degree] = model

    return predictions, metrics, models

# Plot results with confidence intervals
def plot_results(X, y, predictions, title, models):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='gray', label='Data', marker='x')

    for degree, pred in predictions.items():
        model = models[degree]
        if degree == 'sinusoidal':
            plt.plot(X, pred, label=f'Sinusoidal Regression', linewidth=3)
        else:
            plt.plot(X, pred, label=f'Degree {degree}', linewidth=3)

        # Calculate confidence interval
        if degree != 'sinusoidal':
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X[:, np.newaxis])
            y_pred = model.predict(X_poly)
            y_std = np.std([model.predict(X_poly) for _ in range(100)], axis=0)
            plt.fill_between(X, y_pred - y_std, y_pred + y_std, alpha=0.3)

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

# Training and plotting for each dataset
degrees = [1, 2, 3, 'sinusoidal']
titles = ['Linear Trend with Noise', 'Quadratic Trend with Noise', 'Cubic Trend with Noise',
          'Sine Function with Noise', 'Sinusoidal Function with Noise']
metrics_results = {}

for (X, y), title in zip(datasets, titles):
    predictions, metrics, models = train_and_evaluate(X, y, degrees)
    metrics_results[title] = metrics
    print(f'{title}:')
    for degree, metric_values in metrics.items():
        if degree == 'sinusoidal':
            print(f'Sinusoidal Regression:')
        else:
            print(f'Degree {degree}:')
        print(f'  MSE: {metric_values["MSE"]:.4f}')
        print(f'  RMSE: {metric_values["RMSE"]:.4f}')
        print(f'  MAE: {metric_values["MAE"]:.4f}')
        print(f'  R2: {metric_values["R2"]:.4f}')
    plot_results(X, y, predictions, title, models)

metrics_results
