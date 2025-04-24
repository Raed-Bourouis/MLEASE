import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot

def log_eda(data_path):
    """
    Perform exploratory data analysis on a time series dataset and log results to MLflow.
    Generates and logs descriptive statistics, plots, and stationarity tests.
    """
    # Ensure experiment is set (optional)
    mlflow.set_experiment("EDA_TimeSeries")
    # Start an MLflow run
    with mlflow.start_run(run_name="EDA_Run"):
        # Load the dataset
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
        # Data types
        dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
        with open("dtypes.json", "w") as f:
            json.dump(dtypes, f, indent=4)
        mlflow.log_artifact("dtypes.json")
        # Descriptive statistics
        describe = df.describe().to_dict()
        with open("describe.json", "w") as f:
            json.dump(describe, f, indent=4)
        mlflow.log_artifact("describe.json")
        # Missing values
        missing = df.isnull().sum().to_dict()
        with open("missing_values.json", "w") as f:
            json.dump(missing, f, indent=4)
        mlflow.log_artifact("missing_values.json")
        # Correlation matrix and heatmap
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            corr_matrix = df.corr()
            with open("correlation_matrix.json", "w") as f:
                json.dump(corr_matrix.to_dict(), f, indent=4)
            mlflow.log_artifact("correlation_matrix.json")
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig("correlation_heatmap.png")
            mlflow.log_artifact("correlation_heatmap.png")  # Log the heatmap&#8203;:contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
            plt.close()
        # Log skewness and kurtosis as metrics, and generate histograms/boxplots
        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()
            skew_val = float(data.skew())
            kurt_val = float(data.kurtosis())
            mlflow.log_metric(f"skew_{col}", skew_val)
            mlflow.log_metric(f"kurtosis_{col}", kurt_val)
            # Histogram
            plt.figure()
            sns.histplot(data, kde=True)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            hist_file = f"hist_{col}.png"
            plt.savefig(hist_file)
            mlflow.log_artifact(hist_file)  # Log histogram
            plt.close()
            # Boxplot
            plt.figure()
            sns.boxplot(y=data)
            plt.title(f"Boxplot of {col}")
            plt.ylabel(col)
            plt.tight_layout()
            box_file = f"boxplot_{col}.png"
            plt.savefig(box_file)
            mlflow.log_artifact(box_file)  # Log boxplot
            plt.close()
        # Rolling statistics (mean and std over 12 periods)&#8203;:contentReference[oaicite:2]{index=2}
        window = 12
        rolling_means = df.rolling(window=window).mean()
        rolling_stds = df.rolling(window=window).std()
        for col in df.select_dtypes(include=[np.number]).columns:
            plt.figure()
            plt.plot(df.index, rolling_means[col], label='Rolling Mean')
            plt.plot(df.index, rolling_stds[col], label='Rolling Std')
            plt.title(f"Rolling Mean & Std ({window}) for {col}")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.tight_layout()
            roll_file = f"rolling_mean_std_{col}.png"
            plt.savefig(roll_file)
            mlflow.log_artifact(roll_file)  # Log rolling stats
            plt.close()
        # Augmented Dickey-Fuller test for stationarity&#8203;:contentReference[oaicite:3]{index=3}
        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()
            if len(data) > 0:
                adf_result = adfuller(data)
                adf_output = {
                    "Test Statistic": adf_result[0],
                    "p-value": adf_result[1],
                    "used_lag": adf_result[2],
                    "n_obs": adf_result[3],
                    "critical_values": adf_result[4],
                    "ic_best": adf_result[5]
                }
                with open(f"adf_{col}.json", "w") as f:
                    json.dump(adf_output, f, indent=4)
                mlflow.log_artifact(f"adf_{col}.json")  # Log ADF results
        # Seasonal decomposition (trend, seasonal, residual)&#8203;:contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}
        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()
            if len(data) > window:
                try:
                    result = seasonal_decompose(data, model='additive', period=window)
                    # Plot components
                    fig = result.plot()
                    fig.suptitle(f"Seasonal Decompose of {col}", fontsize=16)
                    fig.tight_layout()
                    seasonal_file = f"seasonal_decompose_{col}.png"
                    fig.savefig(seasonal_file)
                    mlflow.log_artifact(seasonal_file)  # Log seasonal decomposition
                    plt.close(fig)
                except Exception:
                    pass
        # ACF and PACF plots&#8203;:contentReference[oaicite:6]{index=6}
        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()
            if len(data) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                plot_acf(data, ax=axes[0], lags=20, title=f'ACF: {col}')
                plot_pacf(data, ax=axes[1], lags=20, title=f'PACF: {col}')
                plt.tight_layout()
                acf_file = f"acf_pacf_{col}.png"
                fig.savefig(acf_file)
                mlflow.log_artifact(acf_file)  # Log ACF and PACF plots
                plt.close(fig)
        # Lag plots&#8203;:contentReference[oaicite:7]{index=7}
        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()
            if len(data) > 1:
                plt.figure()
                lag_plot(data)
                plt.title(f"Lag Plot of {col}")
                lag_file = f"lagplot_{col}.png"
                plt.tight_layout()
                plt.savefig(lag_file)
                mlflow.log_artifact(lag_file)  # Log lag plot
                plt.close()
        # End MLflow run
    print("EDA logging completed.")

if __name__ == "__main__":
    if __name__ == "__main__":
        log_eda("../datasets/BeerWineLiquor.csv")
