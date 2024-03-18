import random
import matplotlib.pyplot as plt
import pandas as pd
from math import sin
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import f1_score

def draw(df):
    plt.scatter(df[df["res"] == 1]["v1"], df[df["res"] == 1]["v2"], marker=".", label="positive(1)")
    plt.scatter(df[df["res"] == -1]["v1"], df[df["res"] == -1]["v2"], marker="^", label="negative(-1)")
    plt.title("Visualization")
    plt.legend()
    plt.show()

def foo_to_classify_logistic_modified(x, y):
    if x > 5 and y < 5:
        return 1 if sin(x / 5) > y / 10 else -1
    elif x < 5 and y > 5:
        return 1 if sin(y / 5) > x / 10 else -1
    else:
        return 1 if sin(x / 5 + y / 5) > (x + y) / 20 - 1 else -1

def generate_dataset(size, func):
    return [[x := random.uniform(0, 10), y := random.uniform(0, 10), func(x, y)] for _ in range(size)]

# Parametry
size_train, size_test = 8000, 2000

# Generowanie danych
dataset_train_modified = generate_dataset(size_train, foo_to_classify_logistic_modified)
dataset_test_modified = generate_dataset(size_test, foo_to_classify_logistic_modified)

# Przygotowanie DataFrame
df_train_modified = pd.DataFrame(dataset_train_modified, columns=["v1", "v2", "res"])
df_test_modified = pd.DataFrame(dataset_test_modified, columns=["v1", "v2", "res"])

# Wizualizacja zmodyfikowanych danych
draw(df_train_modified)

# Przygotowanie danych do modelowania
X_train_modified, y_train_modified = df_train_modified.drop("res", axis=1), df_train_modified["res"]
X_test_modified, y_test_modified = df_test_modified.drop("res", axis=1), df_test_modified["res"]

# Definicja modeli
models = {
    "Linear": LinearRegression(),
    "Logistic": LogisticRegression(random_state=0, max_iter=1000),
    "Ridge": Ridge(random_state=0),
    "Lasso": Lasso(random_state=0)
}

# Trenowanie i ocena modeli
for name, model in models.items():
    model.fit(X_train_modified, y_train_modified)
    y_pred_modified = model.predict(X_test_modified)
    if name == "Logistic":
        y_pred_class_modified = y_pred_modified
    else:
        y_pred_class_modified = np.where(y_pred_modified < 0, -1, 1)
    print(f"{name} (modified): {f1_score(y_test_modified, y_pred_class_modified)}")
