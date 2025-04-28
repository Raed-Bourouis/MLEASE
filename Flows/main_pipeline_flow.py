from prefect import flow, task
import mlflow

# Import your modules
from MLFLOW.Preprocessing import preprocess
from MLFLOW.Model_Select import model_selection_pipeline
from MLFLOW.EDA import log_eda

from MLFLOW.Sarima import train_sarima
from MLFLOW.Prophet import train_prophet
from MLFLOW.Xgboost import train_xgboost
from MLFLOW.lstm import train_lstm
from MLFLOW.Ensemble_combiner import combine_ensemble_predictions

@task
def exploratory_data_analysis(dataset_path):
    print("🔵 Running EDA on raw dataset...")
    log_eda(dataset_path)
    return dataset_path

@task
def data_preprocessing(dataset_path):
    print("🔵 Running preprocessing...")
    preprocessed_path = preprocess(dataset_path)
    return preprocessed_path

@task
def empirical_model_scoring(preprocessed_path):
    print("Running empirical model scoring...")
    model_selection_pipeline(preprocessed_path)

    with open("../MLFLOW/recommended_model.txt", "r") as f:
        selected_model_info = f.read().strip()

    return selected_model_info

@task
def train_selected_model(selected_model_info, dataset_path, target_column):
    if selected_model_info.startswith("ensemble:"):
        models = selected_model_info.split(":")[1].split("+")
        print(f"Training ensemble models: {models}")

        trained_models = []  # To Track models that we trained

        for model in models:
            model = model.lower()
            if model == "sarima":
                train_sarima(dataset_path, target_column, p=1, d=1, q=1, seasonal_order=(1,1,1,12))
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

        # After training exactly these two models, combine their predictions
        combine_ensemble_predictions(trained_models)

    else:
        model = selected_model_info
        print(f"Training single model: {model}")

        if model.lower() == "sarima":
            train_sarima(dataset_path, target_column, p=1, d=1, q=1, seasonal_order=(1,1,1,12))
        elif model.lower() == "prophet":
            train_prophet(dataset_path, target_column, growth='linear')
        elif model.lower() == "xgboost":
            train_xgboost(dataset_path, target_column)
        elif model.lower() == "lstm":
            train_lstm(dataset_path, target_column)
        else:
            raise ValueError(f"Unknown model {model}")

@flow(name="MLEASE Full Pipeline Orchestration")
def full_pipeline_flow(dataset_path: str, target_column: str):
    preprocessed_path = data_preprocessing(dataset_path)
    exploratory_data_analysis(preprocessed_path)
    selected_model_info = empirical_model_scoring(preprocessed_path)
    train_selected_model(selected_model_info, preprocessed_path, target_column)

    print("Full pipeline finished successfully.")

    mlflow.log_param("Pipeline Status", "Completed")
    mlflow.log_param("Selected Model", selected_model_info)

if __name__ == "__main__":
    full_pipeline_flow(
        dataset_path="../datasets/Alcohol_sales.csv",
        target_column="sales"
    )
