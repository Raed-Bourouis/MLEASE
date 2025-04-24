import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
import pandas as pd
import os 
import time


def train_xgboost_model(csv_path, date_col, value_col, output_dir="xgboost_output"):
    mlflow.set_experiment("mlease-training")

    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['time_idx'] = np.arange(len(df))

    X = df[['month', 'year', 'time_idx']]
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run():
        start = time.time()

        params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}
        mlflow.set_tag("model_name", "XGBoost")
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("train_time", time.time() - start)

        os.makedirs(output_dir, exist_ok=True)
        results_df = pd.DataFrame({"ds": df['ds'].iloc[-len(preds):], "y": y_test.values, "yhat": preds})
        results_path = os.path.join(output_dir, "xgboost_forecast.csv")
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        plt.figure(figsize=(10, 4))
        plt.plot(results_df['ds'], results_df['y'], label="Actual")
        plt.plot(results_df['ds'], results_df['yhat'], label="Forecast")
        plt.legend()
        plot_path = os.path.join(output_dir, "xgboost_forecast.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        print(f"[XGBoost] RMSE: {rmse:.4f}")


# Example usage:
if __name__ == "__main__":
    train_xgboost_model(
        csv_path="../datasets/Miles_Traveled.csv",
        date_col="DATE",
        value_col="TRFVOLUSM227NFWA"
    )
