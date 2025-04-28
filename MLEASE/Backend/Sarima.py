import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import mlflow

def train_sarima(df, target_column):
    print(f"Training SARIMA model for {target_column}...")
    
    df.index = pd.to_datetime(df.index)
    series = df[target_column]
    
    split_index = int(len(df) * 0.8)
    train_df = series.iloc[:split_index]
    test_df = series.iloc[split_index:]
    
    with mlflow.start_run(nested=True, run_name=f"SARIMA_{target_column}"):
        auto_model = auto_arima(train_df, seasonal=True, m=12, stepwise=True, trace=True, suppress_warnings=True)
        best_order = auto_model.order
        best_seasonal_order = auto_model.seasonal_order
        
        mlflow.log_param("order", best_order)
        mlflow.log_param("seasonal_order", best_seasonal_order)
        
        model = SARIMAX(train_df, 
                        order=best_order, 
                        seasonal_order=best_seasonal_order,
                        enforce_stationarity=False, 
                        enforce_invertibility=False)
        
        sarima_model = model.fit(disp=False)
        mlflow.sklearn.log_model(sarima_model, "sarima_model")
        
        forecast = sarima_model.forecast(steps=len(test_df))
        mse = mean_squared_error(test_df, forecast)
        
        mlflow.log_metric("mse", mse)
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_df.index, train_df, label="Train")
        plt.plot(test_df.index, test_df, label="Test", color="orange")
        plt.plot(test_df.index, forecast, label="Forecast", linestyle="dashed", color="red")
        plt.title(f"Prédictions SARIMA pour {target_column}")
        plt.legend()
        plt.savefig(f"sarima_forecast_{target_column}.png")
        mlflow.log_artifact(f"sarima_forecast_{target_column}.png")
        plt.close()
        
        forecast_df = pd.DataFrame({"ds": test_df.index, "yhat": forecast.values})
        forecast_df.to_csv(f"forecast_sarima_{target_column}.csv", index=False)
        mlflow.log_artifact(f"forecast_sarima_{target_column}.csv")
    
    return {"mse": mse, "forecast": forecast_df}