from prefect import flow, task, get_run_logger
import mlflow
import pandas as pd
import os
import sys

# Ensure project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom modules
from MLFLOW.Preprocessing import TimeSeriesPreprocessor
from MLFLOW.Model_Select import model_selection_pipeline
from MLFLOW.EDA import log_eda
from MLFLOW.Sarima import train_sarima
from MLFLOW.Prophet import train_prophet
from MLFLOW.Xgboost import train_xgboost
from MLFLOW.lstm import train_lstm
from MLFLOW.Ensemble_combiner import combine_ensemble_predictions

@task
def exploratory_data_analysis(dataset_path: str):
    logger = get_run_logger()
    logger.info("Running EDA on raw dataset...")
    log_eda(dataset_path)
    return dataset_path

@task(name="preprocess_and_log_with_mlflow")
def preprocess(dataset_path: str, eda_report_path: str, output_path: str = "preprocessed_timeseries.csv") -> str:
    logger = get_run_logger()
    mlflow.set_experiment("mlease-preprocessing")

    with mlflow.start_run():
        mlflow.log_param("EDA_report_path", eda_report_path)
        mlflow.log_param("dataset", os.path.basename(dataset_path))

        df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
        preprocessor = TimeSeriesPreprocessor(eda_report_path)
        df_processed = preprocessor.preprocess(df)

        df_processed.to_csv(output_path)
        mlflow.log_artifact(output_path)
        mlflow.log_artifact(eda_report_path)

        logger.info(f"Preprocessing completed. Output saved to {output_path}")
        return output_path

@task
def empirical_model_scoring(preprocessed_path: str) -> str:
    logger = get_run_logger()
    logger.info("Running empirical model scoring...")
    model_selection_pipeline(preprocessed_path)

    with open("../MLFLOW/recommended_model.txt", "r") as f:
        selected_model_info = f.read().strip()

    return selected_model_info

@task
def train_selected_model(selected_model_info: str, dataset_path: str, target_column: str):
    logger = get_run_logger()

    if selected_model_info.startswith("ensemble:"):
        models = selected_model_info.split(":")[1].split("+")
        logger.info(f"Training ensemble models: {models}")
        trained_models = []

        for model in models:
            model = model.lower()
            if model == "sarima":
                train_sarima(dataset_path, target_column, p=1, d=1, q=1, seasonal_order=(1, 1, 1, 12))
                trained_models.append("Sarima")
            elif model == "prophet":
                train_prophet(dataset_path, target_column, growth='linear')
                trained_models.append("Prophet")
            elif model == "xgboost":
                train_xgboost(dataset_path, target_column)
                trained_models.append("Xgboost")
            elif model == "lstm":
                train_lstm(dataset_path, target_column)
                trained_models.append("Lstm")
            else:
                raise ValueError(f"Unknown model {model}")

        combine_ensemble_predictions(trained_models)

    else:
        model = selected_model_info.lower()
        logger.info(f"Training single model: {model}")

        if model == "sarima":
            train_sarima(dataset_path, target_column, p=1, d=1, q=1, seasonal_order=(1, 1, 1, 12))
        elif model == "prophet":
            train_prophet(dataset_path, target_column, growth='linear')
        elif model == "xgboost":
            train_xgboost(dataset_path, target_column)
        elif model == "lstm":
            train_lstm(dataset_path, target_column)
        else:
            raise ValueError(f"Unknown model {model}")

@flow(name="MLEASE Full Pipeline Orchestration")
def full_pipeline_flow(dataset_path: str, eda_report_path: str, target_column: str):
    eda_done_path = exploratory_data_analysis(dataset_path)
    preprocessed_path = preprocess(eda_done_path, eda_report_path)
    selected_model_info = empirical_model_scoring(preprocessed_path)
    train_selected_model(selected_model_info, preprocessed_path, target_column)

    mlflow.set_experiment("mlease-preprocessing")
    with mlflow.start_run():
        mlflow.log_param("Pipeline Status", "Completed")
        mlflow.log_param("Selected Model", selected_model_info)

    print("Full pipeline finished successfully.")

if __name__ == "__main__":
    full_pipeline_flow(
        dataset_path="./datasets/Alcohol_sales.csv",
        eda_report_path="./EDA/mlops_eda_report.json",
        target_column="sales"
    )
