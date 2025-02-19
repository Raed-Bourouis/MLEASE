import json
import os
import pandas as pd
import numpy as np
import concurrent.futures
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_eda_report(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def apply_transformations(df, report, version):
    # Apply stationarity transformations
    for col, details in report['preprocessing_recommendations']['stationarity'].items():
        if details['transformation_needed'] == "YES":
            df[col] = df[col].diff().fillna(df[col])  # Differencing as an example
    
    # Apply feature scaling
    for scaling in report['preprocessing_recommendations']['feature_scaling']:
        col = scaling['column']
        if 'standardization' in scaling['recommended_method']:
            df[col] = StandardScaler().fit_transform(df[[col]])
        elif 'min_max_scaling' in scaling['recommended_method']:
            df[col] = MinMaxScaler().fit_transform(df[[col]])
    
    # Save preprocessed dataset
    output_path = f'preprocessed_dataset_v{version}.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def parallel_preprocessing(dataset_path, report, num_versions=3):
  

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for version in range(1, num_versions + 1):
            future = executor.submit(apply_transformations, df.copy(), report, version)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            future.result()


report_json=load_eda_report("../EDA/mlops_eda_report.json")

df = pd.read_csv('../datasets/Month_Value_1.csv')

# Example usage
parallel_preprocessing(df, report_json, num_versions=3)
