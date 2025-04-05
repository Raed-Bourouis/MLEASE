from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd


def check_stationarity(ts):
    adf_result = adfuller(ts)
    kpss_result, _, _, _ = kpss(ts, regression='c')

    print(f'ADF Test p-value: {adf_result[1]} (Stationary if < 0.05)')
    print(f'KPSS Test p-value: {kpss_result} (Non-Stationary if > 0.05)')
    
    if adf_result[1] < 0.05 and kpss_result < 0.05:
        return "Stationary - Consider SARIMA"
    else:
        return "Non-Stationary - Consider LSTM, XGBoost, Prophet"
    

def decompose_time_series(ts, period=12):
    decomposition = sm.tsa.seasonal_decompose(ts, model='additive', period=period)
    decomposition.plot()
    plt.show()
    
    trend_strength = np.std(decomposition.trend.dropna())
    seasonality_strength = np.std(decomposition.seasonal.dropna())
    
    if trend_strength > seasonality_strength:
        return "Strong Trend - Consider Prophet or LSTM"
    elif seasonality_strength > trend_strength:
        return "Strong Seasonality - Consider SARIMA"
    else:
        return "No Strong Trend/Seasonality - Consider XGBoost"
    

def correlation_analysis(df, target_column):
    correlation_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.show()

    target_corr = correlation_matrix[target_column].sort_values(ascending=False)
    print("Feature Correlation with Target:\n", target_corr)


def check_autocorrelation(ts, lags=40):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(ts, lags=lags, ax=ax[0])
    plot_pacf(ts, lags=lags, ax=ax[1])
    plt.show()

if __name__ == "__main__":
    # Example usage
    ts=pd.read_csv("../preprocessing/output/preprocessed_timeseries0.csv", index_col=0, parse_dates=True)
    target=ts.columns[-1]
    check_stationarity(ts[target])
    decompose_time_series(ts[target])
    correlation_analysis(ts, ts.columns[-1])
    check_autocorrelation(ts[target])