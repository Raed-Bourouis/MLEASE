from prefect import task, get_run_logger
import mlflow
import mlflow.sklearn
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
import warnings
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TimeSeriesPreprocessor:
    def __init__(self, eda_report_path, seasonal_periods=12):
        self.eda_report = self._load_eda_report(eda_report_path)
        self.seasonal_periods = seasonal_periods

    @staticmethod
    def _load_eda_report(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def _check_stationarity(series):
        result = adfuller(series.dropna())
        return result[1] < 0.05

    @staticmethod
    def _apply_transformations(series, method):
        if method == "differencing":
            return series.diff().fillna(series.iloc[0])
        elif method == "log_transformation":
            return np.log1p(series)
        elif method == "rolling_mean_normalization":
            return series - series.rolling(window=3, min_periods=1).mean()
        elif method == "boxcox_transformation":
            return pd.Series(boxcox(series + 1e-6)[0], index=series.index)
        return series
    
    def _handle_missing_values(self, df, validation_fraction=0.1, random_state=42):
        if df.isnull().sum().sum() == 0:
            logging.info("No missing values detected after EDA. Skipping handling.")
            return df

        logging.warning("Missing values detected. Selecting best imputation method.")
        df_validation = df.copy()
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        np.random.seed(random_state)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df_validation[col] = pd.to_numeric(df_validation[col], errors='coerce')
            non_missing_indices = df[df[col].notnull()].index
            n_to_mask = int(len(non_missing_indices) * validation_fraction)
            if n_to_mask > 0:
                masked_indices = np.random.choice(non_missing_indices, n_to_mask, replace=False)
                mask.loc[masked_indices, col] = True
                df_validation.loc[masked_indices, col] = np.nan

        imputed_dfs = {
            "forward_fill": df_validation.fillna(method='ffill'),
            "backward_fill": df_validation.fillna(method='bfill'),
            "mean_imputation": df_validation.fillna(df_validation.mean()),
            "median_imputation": df_validation.fillna(df_validation.median()),
            "interpolation": df_validation.interpolate()
        }

        mse_scores = {}
        for method_name, imputed_df in imputed_dfs.items():
            y_true = df[mask]
            y_pred = imputed_df[mask]
            mse = np.mean((y_true - y_pred) ** 2)
            mse_scores[method_name] = mse

        best_method = min(mse_scores, key=mse_scores.get)
        logging.info(f"Selected best missing value handling method: {best_method}")

        if best_method == "forward_fill":
            return df.fillna(method='ffill')
        elif best_method == "backward_fill":
            return df.fillna(method='bfill')
        elif best_method == "mean_imputation":
            return df.fillna(df.mean())
        elif best_method == "median_imputation":
            return df.fillna(df.median())
        elif best_method == "interpolation":
            return df.interpolate()
        else:
            logging.warning("No valid imputation method selected, returning original df.")
            return df
    
    def _evaluate_transformations(self, series, suggested_methods):
        transformations = {m: self._apply_transformations(series, m) for m in suggested_methods}
        for method, transformed_series in transformations.items():
            if self._check_stationarity(transformed_series):
                logging.info(f"Using recommended transformation: {method}")
                return transformed_series
        
        logging.warning("None of the recommended transformations worked. Using original series.")
        return series
    
    def _scale_data(self, df):
        recommended_methods = self.eda_report["preprocessing_recommendations"]["feature_scaling"][0]["recommended_method"]
        scalers = {"standardization": StandardScaler(), "min_max_scaling": MinMaxScaler(), "robust_scaling": RobustScaler()}
        return df  # Scaling applied later in the pipeline
        for method in recommended_methods:
            if method in scalers:
                logging.info(f"Using recommended scaling: {method}")
                scaler = scalers[method]
                return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        logging.warning("No valid scaling method found. Returning unscaled data.")
        return df

    def preprocess(self, df):
        start_time = time.time()
        logging.info(f"Starting preprocessing. Memory usage: {psutil.virtual_memory().percent}%")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Dataset must have a datetime index.")

        df = self._handle_missing_values(df)
        results = Parallel(n_jobs=-1)(
            delayed(self._evaluate_transformations)(df[col], self.eda_report["preprocessing_recommendations"]["feature_engineering"][i-1]["suggested_transformations"])
            for i, col in enumerate(df.columns)
        )
        df = pd.DataFrame(dict(zip(df.columns, results)), index=df.index)
        # df = self._scale_data(df)  # Optional
        logging.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds. Memory usage: {psutil.virtual_memory().percent}%")
        return df


