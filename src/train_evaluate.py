# Implement and evaluation metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R²": r2
    }


# Example usage
y_true = np.array([10, 20, 30, 40, 50])
y_pred = np.array([12, 18, 33, 37, 55])
metrics = evaluate_forecast(y_true, y_pred)
print(metrics)
# visualize results


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="red", linestyle="--")
    plt.title(f"Actual vs. Predicted Power Consumption ({model_name})")
    plt.xlabel("Time")
    plt.ylabel("Power Consumption")
    plt.legend()
    plt.show()


# Example usage
plot_actual_vs_predicted(y_true, y_pred, "Vanilla Transformer")
# Performance Metrics


# Example metrics for three models and three zones
metrics_data = {
    "Model": ["Vanilla Transformer", "PatchTST", "Other Model"],
    "Zone 1 MAE": [0.5, 0.4, 0.6],
    "Zone 1 RMSE": [0.7, 0.6, 0.8],
    "Zone 2 MAE": [0.6, 0.5, 0.7],
    "Zone 2 RMSE": [0.8, 0.7, 0.9],
    "Zone 3 MAE": [0.4, 0.3, 0.5],
    "Zone 3 RMSE": [0.6, 0.5, 0.7],
}

df_metrics = pd.DataFrame(metrics_data)
print(df_metrics)

# Plotting
df_metrics.set_index("Model").plot(kind="bar", figsize=(
    12, 6), title="Performance Metrics Across Models and Zones")
plt.ylabel("Error")
plt.show()
