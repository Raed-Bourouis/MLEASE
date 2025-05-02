
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import mlflow
import numpy as np
import mlflow.sklearn  
import time
import os

def train_prophet(csv_path, date_col, value_col, model_name="prophet_model", output_dir="./datasets/predictions/"):
    mlflow.set_experiment("mlease-training")

    # Load data
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})

    with mlflow.start_run():
        start = time.time()
        mlflow.set_tag("model_name", model_name)
        mlflow.log_param("framework", "Prophet")

        # Initialize and train Prophet
        model = Prophet()
        model.fit(df)

        # Forecast
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        # Evaluate (on known data)
        y_true = df['y'].values
        y_pred = forecast['yhat'][:len(df)].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("train_time", time.time() - start)

        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        forecast_path = os.path.join(output_dir, "prophet_forecast.csv")
        model_path = os.path.join(output_dir, f"{model_name}.pkl")

        forecast.to_csv(forecast_path, index=False)
        mlflow.log_artifact(forecast_path)

        # Save the model
        import joblib
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"[✓] Prophet training done. RMSE: {rmse:.4f}")

# Example usage
# if __name__ == "__main__":
#     train_prophet_model(
#         csv_path="./datasets/Miles_Traveled.csv",
#         date_col="DATE",
#         value_col="TRFVOLUSM227NFWA"
#     )
