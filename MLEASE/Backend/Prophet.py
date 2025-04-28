import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import mlflow

def train_prophet(df, target_column):
    print(f"Training Prophet model for {target_column}...")
    
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: "ds"}, inplace=True)
    detected_freq = pd.infer_freq(df["ds"]) or "MS"
    
    temp_df = df[["ds", target_column]].rename(columns={target_column: "y"})
    
    split_index = int(len(temp_df) * 0.8)
    train_df = temp_df.iloc[:split_index]
    test_df = temp_df.iloc[split_index:]
    
    with mlflow.start_run(nested=True, run_name=f"Prophet_{target_column}"):
        model = Prophet()
        model.fit(train_df)
        mlflow.sklearn.log_model(model, "prophet_model")
        
        future = model.make_future_dataframe(periods=len(test_df), freq=detected_freq)
        forecast = model.predict(future)
        
        y_true = test_df["y"].values
        y_pred = forecast["yhat"].iloc[-len(test_df):].values
        
        mse = mean_squared_error(y_true, y_pred)
        mlflow.log_metric("mse", mse)
        
        plt.figure(figsize=(10, 5))
        plt.plot(temp_df["ds"], temp_df["y"], label="Données Réelles", alpha=0.7)
        plt.plot(forecast["ds"], forecast["yhat"], label="Prédictions", linestyle="dashed")
        plt.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="gray",
            alpha=0.2,
        )
        plt.title(f"Prévisions pour {target_column}")
        plt.legend()
        plt.savefig(f"prophet_forecast_{target_column}.png")
        mlflow.log_artifact(f"prophet_forecast_{target_column}.png")
        plt.close()
        
        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        forecast_df.to_csv(f"forecast_prophet_{target_column}.csv", index=False)
        mlflow.log_artifact(f"forecast_prophet_{target_column}.csv")
    
    return {"mse": mse, "forecast": forecast_df}