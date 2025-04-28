# ensemble_combiner.py

import pandas as pd
import mlflow
import os

def combine_ensemble_predictions(models_list, output_path="../datasets/ensemble_predictions.csv"):
    """
    Combine predictions from selected models using simple average.
    
    Args:
        models_list (list): List of model names, e.g., ["Prophet", "Xgboost"]
        output_path (str): Path where to save the ensemble predictions.
    """
    print(f"🔵 Starting combination of models: {models_list}")

    prediction_dataframes = []

    for model in models_list:
        model = model.lower()
        
        if model == "prophet":
            prediction_file = "../MLFLOW/predictions_prophet.csv"
        elif model == "xgboost":
            prediction_file = "../MLFLOW/predictions_xgboost.csv"
        elif model == "sarima":
            prediction_file = "../MLFLOW/predictions_sarima.csv"
        elif model == "lstm":
            prediction_file = "../MLFLOW/predictions_lstm.csv"
        else:
            raise ValueError(f"Unknown model {model} for ensembling.")

        # Check if file exists
        if not os.path.exists(prediction_file):
            raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

        # Load the predictions
        df = pd.read_csv(prediction_file)

        # Check the required column
        if "yhat" not in df.columns:
            raise ValueError(f"'yhat' column not found in {prediction_file}")

        prediction_dataframes.append(df["yhat"].values)

    # Simple average of predictions
    combined_predictions = sum(prediction_dataframes) / len(prediction_dataframes)

    # Save combined predictions
    ensemble_df = pd.DataFrame({
        "yhat_ensemble": combined_predictions
    })

    ensemble_df.to_csv(output_path, index=False)
    print(f"✅ Ensemble predictions saved to {output_path}")

    # Log to MLflow
    with mlflow.start_run(run_name="Ensemble_Combination"):
        mlflow.log_param("Ensemble Models", "+".join(models_list))
        mlflow.log_artifact(output_path)

    print("✅ Ensemble predictions logged to MLflow.")

# Standalone usage
if __name__ == "__main__":
    example_models = ["Prophet", "Xgboost"]  # Example usage
    combine_ensemble_predictions(example_models)
