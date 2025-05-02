import pandas as pd
import numpy as np
import mlflow
import time
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


def train_sarima(csv_path, date_col, value_col, order, seasonal_order, output_dir="./datasets/predictions/"):
    mlflow.set_experiment("mlease-training")

    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    df.set_index("ds", inplace=True)

    with mlflow.start_run():
        start = time.time()

        mlflow.set_tag("model_name", "SARIMA")
        mlflow.log_param("order", order)
        mlflow.log_param("seasonal_order", seasonal_order)

        model = SARIMAX(df['y'], order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)

        forecast = results.get_prediction(start=0, end=len(df)-1)
        y_pred = forecast.predicted_mean
        y_true = df['y']

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("train_time", time.time() - start)

        os.makedirs(output_dir, exist_ok=True)
        forecast_df = pd.DataFrame({"ds": df.index, "y": y_true, "yhat": y_pred})
        forecast_path = os.path.join(output_dir, "sarima_forecast.csv")
        forecast_df.to_csv(forecast_path, index=False)
        mlflow.log_artifact(forecast_path)

        plt.figure(figsize=(10, 4))
        plt.plot(df.index, y_true, label="Actual")
        plt.plot(df.index, y_pred, label="Forecast")
        plt.legend()
        plot_path = os.path.join(output_dir, "sarima_forecast.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        print(f"[SARIMA] RMSE: {rmse:.4f}")
        
# if __name__ == "__main__":
#     train_sarima_model(
#         csv_path="./datasets/Miles_Traveled.csv",
#         date_col="DATE",
#         value_col="TRFVOLUSM227NFWA",
#         order=(1, 1, 1),
#         seasonal_order=(1, 1, 1, 12)
#     )


