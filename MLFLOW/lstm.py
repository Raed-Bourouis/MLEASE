# Import des bibliothèques nécessaires
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def train_lstm_pmodel(
    df: pd.DataFrame,
    experiment_name: str = "LSTM_Experiment",
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 0.001
):
    """
    Pipeline d'entraînement LSTM pour une prévision multivariée.

    Paramètres :
        df (pd.DataFrame) : Données temporelles indexées.
        experiment_name (str) : Nom de l'expérience dans MLflow.
        epochs (int) : Nombre d'epochs pour l'entraînement.
        batch_size (int) : Taille des batchs pour l'entraînement.
        learning_rate (float) : Taux d'apprentissage pour l'optimiseur Adam.
    """

    # 1. Préparer les données
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)

    # Identifier les colonnes cibles (toutes sauf la date)
    target_columns = df.columns.tolist()
    forecast_results = {}

    # 2. Normalisation des données
    # LSTM fonctionne mieux avec des données normalisées entre 0 et 1
    scalers = {}
    scaled_data = {}

    for column in target_columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[column] = scaler.fit_transform(df[[column]])
        scalers[column] = scaler

    # 3. Transformer les données en séquences pour LSTM
    def create_sequences(data, seq_length=12):
        """
        Transforme les données en séquences pour le modèle LSTM.
        Chaque séquence contient seq_length valeurs passées pour prédire la suivante.
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    # Déterminer automatiquement la fréquence et adapter le nombre de pas
    detected_freq = pd.infer_freq(df.index)
    print("***", detected_freq, "***")
    if detected_freq in ["D", "B"]:  # Quotidienne ou Business Days
        seq_length = 10
    elif detected_freq in ["MS", "M"]:  # Mensuelle
        seq_length = 12
    else:
        seq_length = 1
    print(seq_length)

    # Générer les séquences pour chaque variable cible
    sequences = {}
    for column in target_columns:
        X, y = create_sequences(scaled_data[column], seq_length)
        sequences[column] = (X, y)

    # 4. Séparer les données en Train (80%) et Test (20%)
    train_size = int(0.8 * len(df))
    train_sequences = {}
    test_sequences = {}

    for column in target_columns:
        X, y = sequences[column]
        train_sequences[column] = (X[:train_size], y[:train_size])
        test_sequences[column] = (X[train_size:], y[train_size:])

    # 5. Définir l'architecture du modèle LSTM
    def build_lstm_model(input_shape):
        """
        Construit un modèle LSTM avec deux couches cachées.
        """
        model = Sequential([
            LSTM(200, activation='relu', return_sequences=True, input_shape=input_shape),
            LSTM(200, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model

    # 6. Démarrer l'expérience MLflow
    mlflow.set_experiment(experiment_name)

    # 7. Entraîner un modèle LSTM pour chaque colonne
    for column in target_columns:
        with mlflow.start_run(run_name=f"LSTM_{column}"):
            print(f"Training LSTM model for {column}...")

            X_train, y_train = train_sequences[column]
            X_test, y_test = test_sequences[column]

            # Construire et entraîner le modèle
            model = build_lstm_model((seq_length, 1))
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

            # Évaluer la performance sur le jeu de train et test
            train_loss = model.evaluate(X_train, y_train, verbose=0)
            test_loss = model.evaluate(X_test, y_test, verbose=0)

            # Faire des prédictions
            y_pred = model.predict(X_test)

            # Inverser la normalisation
            y_pred_rescaled = scalers[column].inverse_transform(y_pred)
            y_test_rescaled = scalers[column].inverse_transform(y_test)

            # Calculer les erreurs
            mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
            r2 = r2_score(y_test_rescaled, y_pred_rescaled)
            mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)

            # 8. Suivi avec MLflow
            # Enregistrer les hyperparamètres
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("seq_length", seq_length)
            mlflow.log_param("layers", 2)

            # Enregistrer les métriques
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mape", mape)

            # Enregistrer le modèle
            mlflow.keras.log_model(model, artifact_path=f"model_{column}")

            # Visualiser et sauvegarder les prévisions
            plt.figure(figsize=(10, 5))
            plt.plot(df.index[:train_size], df[column][:train_size], label="Train")
            plt.plot(df.index[train_size:], df[column][train_size:], label="Test", color="orange")
            plt.plot(df.index[train_size + seq_length:], y_pred_rescaled, label="Forecast", linestyle="dashed", color="red")
            plt.title(f"Prédictions LSTM pour {column}")
            plt.legend()

            # Sauvegarder localement le graphique
            os.makedirs("plots", exist_ok=True)
            plot_path = f"plots/forecast_{column}.png"
            plt.savefig(plot_path)
            plt.close()

            # Enregistrer le graphique dans MLflow
            mlflow.log_artifact(plot_path)

            # Sauvegarder les prévisions
            forecast_results[column] = pd.DataFrame({
                "ds": df.index[train_size + seq_length:],
                "yhat": y_pred_rescaled.flatten()
            })

    # 9. Exporter toutes les prévisions localement
    for col, forecast in forecast_results.items():
        forecast.to_csv(f"forecast_lstm_{col}.csv", index=False)

    print("Toutes les prédictions LSTM ont été enregistrées avec suivi MLflow.")
