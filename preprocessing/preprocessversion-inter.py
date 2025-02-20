import json
import pandas as pd
import numpy as np
import logging
import time
import psutil
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from scipy.stats import boxcox
from joblib import Parallel, delayed

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TimeSeriesPreprocessor:
    def __init__(self, eda_report_path, seasonal_periods=12):
        """
        Initializes the time series preprocessor.
        
        Parameters:
            eda_report_path (str): Path to the JSON file (rapport de l'EDA).
            seasonal_periods (int): The seasonal lag used for seasonal differencing.
        """
        self.eda_report = self._load_eda_report(eda_report_path)
        self.seasonal_periods = seasonal_periods

    def _load_eda_report(self, json_file):
        """Loads the EDA report JSON file."""
        with open(json_file, 'r') as f:
            return json.load(f)
    
    def _check_stationarity(self, series):
        """pour Retester au cas où il y avait un problème."""
        result = adfuller(series.dropna())
        return result[1] < 0.05  # Returns True if stationary

    def _apply_differencing(self, series):
        return series.diff().fillna(series.iloc[0])

    def _apply_log_transformation(self, series):
        return np.log1p(series)

    def _apply_rolling_mean_normalization(self, series, window=3):
        return series - series.rolling(window=window, min_periods=1).mean()
    
    def _apply_boxcox_transformation(self, series):
        return pd.Series(boxcox(series + 1e-6)[0], index=series.index)
    
    def _handle_missing_values(self, df):
        """Verifies if EDA's missing value handling was effective. If not, applies an optimized method."""
        
        # Step 1: Check if missing values still exist
        if df.isnull().sum().sum() == 0:
            logging.info("No missing values detected after EDA. Skipping missing value handling.")
            return df
        
        logging.warning("Missing values detected post-EDA. Applying additional handling.")

        # Step 2: Define multiple imputation strategies
        methods = {
            "forward_fill": df.fillna(method='ffill'),
            "backward_fill": df.fillna(method='bfill'),
            "mean_imputation": df.fillna(df.mean()),
            "median_imputation": df.fillna(df.median()),
            "mode_imputation": df.apply(lambda col: col.fillna(col.mode()[0]) if not col.mode().empty else col, axis=0),
            "interpolation": df.interpolate()
        }
        
        # Step 3: Select the best imputation strategy based on MSE
        best_method = "forward_fill"
        best_mse = float("inf")

        for method, transformed_df in methods.items():
            mse = mean_squared_error(df.dropna(), transformed_df.dropna())
            if mse < best_mse:
                best_mse = mse
                best_method = method
        
        logging.info(f"Selected best missing value handling method: {best_method}")
        return methods[best_method]
    
    def _evaluate_transformations(self, series, suggested_methods):
        transformations = {
            "original": series,
            "differencing": self._apply_differencing(series),
            "log_transformation": self._apply_log_transformation(series),
            "rolling_mean_normalization": self._apply_rolling_mean_normalization(series),
            "boxcox_transformation": self._apply_boxcox_transformation(series)
        }
        
        for method in suggested_methods:
            if method in transformations and self._check_stationarity(transformations[method]):
                logging.info(f"Using recommended transformation: {method}")
                return method, transformations[method]
        
        logging.warning(f"Recommended transformations {suggested_methods} were not effective. Searching for best alternative.")
        
        best_method = "original"
        best_mse = float("inf")
        for method, transformed_series in transformations.items():
            if self._check_stationarity(transformed_series):
                mse = mean_squared_error(series.dropna(), transformed_series.dropna())
                if mse < best_mse:
                    best_mse = mse
                    best_method = method
        return best_method, transformations[best_method]
    
    def _scale_data(self, df):
        recommended_methods = self.eda_report["preprocessing_recommendations"]["feature_scaling"][0]["recommended_method"]
        scaling_methods = {
            "standardization": StandardScaler(),
            "min_max_scaling": MinMaxScaler(),
            "robust_scaling": RobustScaler()
        }
        
        for method in recommended_methods:
            if method in scaling_methods:
                logging.info(f"Using recommended scaling method: {method}")
                scaler = scaling_methods[method]
                return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        
        logging.warning("No recommended scaling method was applicable. Using default.")
        return df
    
    def preprocess(self, df):
        start_time = time.time()
        logging.info(f"Starting preprocessing. Initial memory usage: {psutil.virtual_memory().percent}%")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Dataset must have a datetime index.")
        
        df = self._handle_missing_values(df)
        
        columns = df.columns
        results = Parallel(n_jobs=-1)(delayed(self._evaluate_transformations)(df[col], self.eda_report["preprocessing_recommendations"]["feature_engineering"][i]["suggested_transformations"]) for i, col in enumerate(columns))
        
        for (col, (best_method, best_series)) in zip(columns, results):
            logging.info(f"Best transformation for {col}: {best_method}")
            df[col] = best_series
        
        df = self._scale_data(df)
        
        elapsed_time = time.time() - start_time
        logging.info(f"Preprocessing completed in {elapsed_time:.2f} seconds. Final memory usage: {psutil.virtual_memory().percent}%")
        return df

if __name__ == "__main__":
    eda_report_path = "EDA/mlops_eda_report.json"
    df = pd.read_csv("preprocessing/df.Csv", index_col=0, parse_dates=True)
    
    preprocessor = TimeSeriesPreprocessor(eda_report_path)
    df_processed = preprocessor.preprocess(df)
    
    df_processed.to_csv("preprocessed_timeseries.csv")
    print("Preprocessing finished. Data saved.")
