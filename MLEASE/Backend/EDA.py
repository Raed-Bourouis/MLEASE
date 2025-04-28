import warnings
warnings.filterwarnings('ignore')
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ydata_profiling as yd
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pandas.api.types import is_datetime64_any_dtype
import mlflow

def handle_missing_values(df, validation_fraction=0.1, random_state=42):
    if df.isnull().sum().sum() == 0:
        print("No missing values detected after EDA. Skipping handling.")
        return df
    print("Missing values detected. Selecting best imputation method.")
    
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
    print(mse_scores)
    best_method = min(mse_scores, key=mse_scores.get)
    print(f"Selected best missing value handling method: {best_method}")

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
        print("No valid imputation method selected, returning original df.")
        return df

def generate_mlops_report(df):
    report = {
        "dataset_metadata": {
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max())
            }
        },
        "preprocessing_recommendations": {
            "stationarity": {},
            "feature_scaling": [],
            "feature_engineering": []
        },
        "statistical_insights": {
            "descriptive_stats": {},
            "correlations": {
                "significant_correlations": [],
                "correlation_matrix": {}
            }
        }
    }
    
    for column in df.columns:
        adf_result = adfuller(df[column])
        report["preprocessing_recommendations"]["stationarity"][column] = {
            "is_stationary": str(adf_result[1] < 0.05),
            "p_value": float(adf_result[1]),
            "transformation_needed": "YES" if adf_result[1] >= 0.05 else "NO"
        }
        
        if adf_result[1] >= 0.05:
            report["preprocessing_recommendations"]["feature_engineering"].append({
                "column": column,
                "suggested_transformations": [
                    "log_transformation",
                    "differencing",
                    "rolling_mean_normalization"
                ]
            })
    
    for column in df.columns:
        report["statistical_insights"]["descriptive_stats"][column] = {
            "mean": float(df[column].mean()),
            "std": float(df[column].std()),
            "min": float(df[column].min()),
            "max": float(df[column].max())
        }
    
    corr_matrix = df.corr()
    report["statistical_insights"]["correlations"]["correlation_matrix"] = \
        {str(col): {str(subcol): float(corr_matrix.loc[col, subcol]) 
                    for subcol in df.columns} 
         for col in df.columns}
    
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                correlation = float(corr_matrix.loc[col1, col2])
                if abs(correlation) > 0.5:
                    report["statistical_insights"]["correlations"]["significant_correlations"].append({
                        "features": [col1, col2],
                        "correlation": correlation,
                        "strength": "strong" if abs(correlation) > 0.7 else "moderate"
                    })
    
    for column in df.columns:
        if df[column].std() > 1:
            report["preprocessing_recommendations"]["feature_scaling"].append({
                "column": column,
                "recommended_method": ["standardization", "min_max_scaling"]
            })
    
    return report

def run_eda(df, output_report_path):
    # Détecter et définir une colonne datetime comme index
    for col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if is_datetime64_any_dtype(df[col]):
            df.set_index(col, inplace=True)
            print(f"Set '{col}' as the datetime index.")
            break 
    
    # Gérer les valeurs manquantes
    mlflow.log_metric("missing_values_before", df.isnull().sum().sum())
    df = handle_missing_values(df)
    mlflow.log_metric("missing_values_after", df.isnull().sum().sum())

    # Générer le rapport MLOps
    final_report = generate_mlops_report(df)
    with open(output_report_path, 'w') as f:
        json.dump(final_report, f, indent=4)
    mlflow.log_artifact(output_report_path)

    # Visualisations (sauvegardées comme artefacts)
    for column in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[column], label=column)
        plt.title(f'Time Series Plot for {column}')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.savefig(f"time_series_{column}.png")
        mlflow.log_artifact(f"time_series_{column}.png")
        plt.close()

    for column in df.columns:
        plt.figure(figsize=(10, 5))
        decomposition = seasonal_decompose(df[column], model='additive', period=12)
        decomposition.plot()
        plt.suptitle(f'Seasonal Decomposition of {column}')
        plt.savefig(f"decomposition_{column}.png")
        mlflow.log_artifact(f"decomposition_{column}.png")
        plt.close()

    print("EDA completed and report saved.")