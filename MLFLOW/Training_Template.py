import mlflow
import time
import os

from sklearn.metrics import mean_squared_error
import joblib  # or use your framework's save method

def train_model(model, X_train, y_train, X_test, y_test, params, model_name, save_path="model_output"):
    """
    Trains a model and logs the run to MLflow.

    Parameters:
    - model: the initialized ML model (sklearn, XGBoost, LSTM wrapper, etc.)
    - X_train, y_train: training data
    - X_test, y_test: testing data
    - params: dict of parameters to log
    - model_name: name to display in MLflow
    - save_path: path to save the model and outputs
    """

    mlflow.set_experiment("mlease-training")

    with mlflow.start_run():
        start_time = time.time()
        mlflow.set_tag("model_name", model_name)

        # Log hyperparameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Train the model
        model.fit(X_train, y_train)

        # Predict and evaluate
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mlflow.log_metric("rmse", rmse)

        duration = time.time() - start_time
        mlflow.log_metric("train_time", duration)

        # Save model
        os.makedirs(save_path, exist_ok=True)
        model_file = os.path.join(save_path, f"{model_name}_model.pkl")
        joblib.dump(model, model_file)
        mlflow.log_artifact(model_file)

        # Optional: log plots, config files, etc.
        # mlflow.log_artifact("your_plot.png")

        print(f"{model_name} training complete. RMSE: {rmse:.4f} | Time: {duration:.2f}s")
