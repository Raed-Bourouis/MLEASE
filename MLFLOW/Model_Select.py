import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import warnings
import mlflow
warnings.filterwarnings('ignore')

def analyze_timeseries(df, target_col=None, datetime_col=None):
    """
    Analyze a timeseries dataset and recommend the most suitable model or ensemble.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The timeseries dataset
    target_col : str, optional
        The target column to analyze. If None and df is a Series or single-column DataFrame, 
        will use that column.
    datetime_col : str, optional
        The column containing datetime information. If None, assumes the DataFrame index is datetime.
    
    Returns:
    --------
    dict : Recommendation results including model or ensemble suggestion with explanation
    """
    # Preprocessing to ensure we have a proper time series
    if isinstance(df, pd.Series):
        series = df
        df = pd.DataFrame(df)
        target_col = df.columns[0]
    elif target_col is None and df.shape[1] == 1:
        target_col = df.columns[0]
        series = df[target_col]
    elif target_col is not None:
        series = df[target_col]
    else:
        raise ValueError("For multivariate data, please specify the target column.")
    
    # Ensure datetime index
    if datetime_col is not None:
        df = df.set_index(datetime_col)
    
    # Check if index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            raise ValueError("Index must be convertible to datetime. Please provide a datetime_col.")
    
    # Create dictionary to store analysis results
    analysis = {}
    
    # Basic dataset properties
    analysis['length'] = len(df)
    analysis['variables'] = df.shape[1]
    analysis['missing_values'] = df.isnull().sum().sum()
    
    # Check for stationarity using ADF test
    adf_result = adfuller(series.dropna())
    analysis['stationary'] = adf_result[1] < 0.05  # p-value < 0.05 means stationary
    
    # Check for trend and seasonality
    try:
        # Frequency detection
        if df.index.freq is None:
            # Try to infer frequency
            df = df.asfreq(pd.infer_freq(df.index))
        
        decomposition = seasonal_decompose(series.dropna(), model='additive')
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        residual = decomposition.resid.dropna()
        
        analysis['trend_strength'] = np.std(trend) / (np.std(residual) + np.std(trend))
        analysis['seasonality_strength'] = np.std(seasonal) / (np.std(residual) + np.std(seasonal))
        analysis['has_trend'] = analysis['trend_strength'] > 0.3
        analysis['has_seasonality'] = analysis['seasonality_strength'] > 0.3
    except:
        analysis['has_trend'] = None
        analysis['has_seasonality'] = None
    
    # Check for linearity (using skewness and kurtosis)
    analysis['skewness'] = skew(series.dropna())
    analysis['kurtosis'] = kurtosis(series.dropna())
    analysis['linear'] = abs(analysis['skewness']) < 1 and abs(analysis['kurtosis']) < 3
    
    # Check for autocorrelation
    try:
        acf_values = acf(series.dropna(), nlags=min(50, len(series)//4))
        analysis['significant_autocorrelation'] = any(abs(acf_values[1:]) > 1.96/np.sqrt(len(series)))
    except:
        analysis['significant_autocorrelation'] = None
    
    # Complex patterns detection
    analysis['complex_patterns'] = not analysis.get('linear', True) or analysis['variables'] > 3
    
    # Dataset size categorization
    if analysis['length'] < 100:
        analysis['size'] = 'small'
    elif analysis['length'] < 1000:
        analysis['size'] = 'medium'
    else:
        analysis['size'] = 'large'
    
    # Time frequency detection
    try:
        if df.index.freq is None:
            freq = pd.infer_freq(df.index)
        else:
            freq = df.index.freq
        analysis['frequency'] = freq
        
        # Determine if high frequency data
        if freq in ['H', 'min', 'T', 'S']:
            analysis['high_frequency'] = True
        else:
            analysis['high_frequency'] = False
    except:
        analysis['frequency'] = None
        analysis['high_frequency'] = False
    
    # Make recommendation based on the analysis
    recommendation, explanation, model_scores = recommend_model(analysis)
    
    # Check if ensemble would be beneficial
    ensemble_recommendation, ensemble_explanation = consider_ensemble(model_scores, analysis)
    
    if ensemble_recommendation:
        return {
            'recommended_approach': 'ensemble',
            'ensemble_details': ensemble_recommendation,
            'explanation': ensemble_explanation,
            'individual_model_scores': model_scores,
            'analysis': analysis
        }
    else:
        return {
            'recommended_approach': 'single_model',
            'recommended_model': recommendation,
            'explanation': explanation,
            'model_scores': model_scores,
            'analysis': analysis
        }

def recommend_model(analysis):
    """
    Recommend a model based on analysis results.
    
    Parameters:
    -----------
    analysis : dict
        Dictionary containing analysis results
    
    Returns:
    --------
    tuple : (recommended_model, explanation, model_scores)
    """
    
    explanations = []
    scores = {
        'SARIMA': 0,
        'Prophet': 0,
        'XGBoost': 0,
        'LSTM': 0
    }
    
    model_explanations = {}
    
    # SARIMA strengths
    if analysis.get('stationary', False) or analysis.get('has_seasonality'):
        scores['SARIMA'] += 2
        model_explanations.setdefault('SARIMA', []).append("Handles stationarity or seasonality well")
    if analysis.get('significant_autocorrelation', False):
        scores['SARIMA'] += 2
        model_explanations.setdefault('SARIMA', []).append("Designed for significant autocorrelation")
    if analysis.get('size') == 'small' or analysis.get('size') == 'medium':
        scores['SARIMA'] += 1
        model_explanations.setdefault('SARIMA', []).append("Works well with small/medium datasets")
    if analysis.get('variables', 1) > 1:
        scores['SARIMA'] -= 2
        model_explanations.setdefault('SARIMA', []).append("Less effective with multiple variables")
    if analysis.get('high_frequency', False):
        scores['SARIMA'] -= 1
        model_explanations.setdefault('SARIMA', []).append("Can be computationally intensive for high-frequency data")
    
    # Prophet strengths
    if analysis.get('has_seasonality', False):
        scores['Prophet'] += 3
        model_explanations.setdefault('Prophet', []).append("Excels at modeling seasonality")
    if analysis.get('has_trend', False):
        scores['Prophet'] += 2
        model_explanations.setdefault('Prophet', []).append("Handles trends effectively")
    if analysis.get('missing_values', 0) > 0:
        scores['Prophet'] += 2
        model_explanations.setdefault('Prophet', []).append("Handles missing values automatically")
    if analysis.get('variables', 1) > 2:
        scores['Prophet'] -= 1
        model_explanations.setdefault('Prophet', []).append("Primarily designed for univariate forecasting with regressors")
    if analysis.get('complex_patterns', True) and not analysis.get('has_seasonality', False) and not analysis.get('has_trend', False):
        scores['Prophet'] -= 1
        model_explanations.setdefault('Prophet', []).append("May miss complex non-seasonal patterns")
    
    # XGBoost strengths
    if analysis.get('complex_patterns', False):
        scores['XGBoost'] += 2
        model_explanations.setdefault('XGBoost', []).append("Models complex patterns effectively")
    if not analysis.get('linear', True):
        scores['XGBoost'] += 2
        model_explanations.setdefault('XGBoost', []).append("Handles non-linear relationships well")
    if analysis.get('variables', 1) > 1:
        scores['XGBoost'] += 2
        model_explanations.setdefault('XGBoost', []).append("Effectively utilizes multiple variables")
    if analysis.get('size') == 'small':
        scores['XGBoost'] -= 1
        model_explanations.setdefault('XGBoost', []).append("May need more data to learn patterns effectively")
    if analysis.get('has_seasonality', False) and not analysis.get('variables', 1) > 1:
        scores['XGBoost'] -= 1
        model_explanations.setdefault('XGBoost', []).append("May need feature engineering to capture seasonality")
    
    # LSTM strengths
    if analysis.get('complex_patterns', False):
        scores['LSTM'] += 3
        model_explanations.setdefault('LSTM', []).append("Excels at modeling complex temporal patterns")
    if analysis.get('size') == 'large':
        scores['LSTM'] += 2
        model_explanations.setdefault('LSTM', []).append("Performs well with large datasets")
    if analysis.get('variables', 1) > 1:
        scores['LSTM'] += 2
        model_explanations.setdefault('LSTM', []).append("Effectively utilizes multiple variables")
    if analysis.get('size') == 'small':
        scores['LSTM'] -= 3
        model_explanations.setdefault('LSTM', []).append("Needs substantial data for training")
    elif analysis.get('size') == 'medium':
        scores['LSTM'] -= 1
        model_explanations.setdefault('LSTM', []).append("Benefits from more training data")
    if analysis.get('has_seasonality', False) and not analysis.get('complex_patterns', False):
        scores['LSTM'] -= 1
        model_explanations.setdefault('LSTM', []).append("May be unnecessarily complex for simple seasonal patterns")
    
    # Select the model with the highest score
    recommended_model = max(scores, key=scores.get)
    
    # Generate specific recommendation explanation
    model_descriptions = {
        'SARIMA': "SARIMA is recommended for this dataset due to its ability to handle time series with seasonality and autocorrelation. It's particularly effective for datasets with clear seasonal patterns.",
        'Prophet': "Prophet is recommended for this dataset as it excels at modeling time series with multiple seasonal patterns and trends. It's robust to missing data and outliers.",
        'XGBoost': "XGBoost is recommended for this dataset due to its effectiveness with non-linear relationships and multiple variables. It can capture complex patterns in the data.",
        'LSTM': "LSTM is recommended for this dataset due to its ability to model complex temporal dependencies and non-linear patterns. It's particularly suitable for large datasets with multiple variables."
    }
    
    # Add model-specific explanations that contributed to the decision
    specific_explanation = model_descriptions[recommended_model] + "\n\nKey factors in this recommendation:"
    for point in model_explanations.get(recommended_model, []):
        specific_explanation += f"\n- {point}"
    
    return recommended_model, specific_explanation, scores

def consider_ensemble(model_scores, analysis):
    """
    Determine if an ensemble approach would be beneficial based on model scores
    and dataset characteristics.
    
    Parameters:
    -----------
    model_scores : dict
        Dictionary of scores for each model
    analysis : dict
        Dictionary containing analysis results
    
    Returns:
    --------
    tuple : (ensemble_recommendation, explanation)
        ensemble_recommendation is None if no ensemble is recommended,
        otherwise it's a dictionary with ensemble details
    """
    # Sort models by their scores
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    top_model, top_score = sorted_models[0]
    runner_up, runner_up_score = sorted_models[1]
    
    # Initialize ensemble recommendation
    ensemble_recommendation = None
    explanation = ""
    
    # Check if scores are close (within 20% of each other)
    score_difference_ratio = (top_score - runner_up_score) / max(top_score, 1)
    
    # Conditions that favor ensembling
    ensemble_favored = False
    reasons = []
    
    # If top models have similar scores
    if score_difference_ratio < 0.2:
        ensemble_favored = True
        reasons.append(f"The top models ({top_model} and {runner_up}) have similar performance scores, suggesting they capture different aspects of the data.")
    
    # If dataset is complex
    if analysis.get('complex_patterns', False):
        ensemble_favored = True
        reasons.append("The dataset exhibits complex patterns that could benefit from multiple modeling approaches.")
    
    # If dataset is large enough
    if analysis.get('size') == 'medium' or analysis.get('size') == 'large':
        ensemble_favored = True
        reasons.append("The dataset is large enough to support training multiple models effectively.")
    
    # If there's both seasonality and complex patterns
    if analysis.get('has_seasonality', False) and analysis.get('complex_patterns', False):
        ensemble_favored = True
        reasons.append("The data shows both seasonality and complex patterns, which different models may capture differently.")
    
    # If has multiple variables
    if analysis.get('variables', 1) > 2:
        ensemble_favored = True
        reasons.append("The multivariate nature of the data suggests different models may excel at capturing different variable relationships.")
    
    # Determine best ensemble combination based on dataset characteristics
    if ensemble_favored and len(reasons) >= 2:  # Require at least 2 reasons to recommend ensemble
        # Identify complementary models
        ensemble_pairs = []
        
        # SARIMA + XGBoost: Good for seasonal data with complex patterns
        if 'SARIMA' in [top_model, runner_up] and 'XGBoost' in [top_model, runner_up]:
            ensemble_pairs.append(('SARIMA', 'XGBoost', "SARIMA captures seasonal and linear components while XGBoost models non-linear relationships"))
        
        # Prophet + XGBoost: Good for trend/seasonal + complex relationships
        if 'Prophet' in [top_model, runner_up] and 'XGBoost' in [top_model, runner_up]:
            ensemble_pairs.append(('Prophet', 'XGBoost', "Prophet handles trends and seasonality while XGBoost captures non-linear relationships"))
        
        # LSTM + SARIMA: Deep learning with statistical approach
        if 'LSTM' in [top_model, runner_up] and 'SARIMA' in [top_model, runner_up]:
            ensemble_pairs.append(('LSTM', 'SARIMA', "LSTM models complex temporal patterns while SARIMA provides statistical rigor"))
        
        # Prophet + LSTM: Good for structured seasonality + complex patterns
        if 'Prophet' in [top_model, runner_up] and 'LSTM' in [top_model, runner_up]:
            ensemble_pairs.append(('Prophet', 'LSTM', "Prophet decomposes seasonality while LSTM captures complex dependencies"))
        
        # If no specific pairs are in top two, suggest based on dataset
        if not ensemble_pairs:
            if analysis.get('has_seasonality', False) and analysis.get('complex_patterns', False):
                ensemble_pairs.append(('Prophet', 'XGBoost', "Prophet for seasonality and XGBoost for complex patterns"))
            elif analysis.get('has_seasonality', False) and analysis.get('size') == 'large':
                ensemble_pairs.append(('SARIMA', 'LSTM', "SARIMA for statistical modeling of seasonality and LSTM for complex patterns"))
            elif analysis.get('variables', 1) > 2 and not analysis.get('has_seasonality', False):
                ensemble_pairs.append(('XGBoost', 'LSTM', "XGBoost and LSTM both handle multivariate data with different approaches"))
            else:
                # If no clear pair, use top two models
                ensemble_pairs.append((top_model, runner_up, f"Combining the strengths of the top two models ({top_model} and {runner_up})"))
        
        # Choose the best ensemble pair
        if ensemble_pairs:
            model1, model2, pair_reason = ensemble_pairs[0]
            
            # Create ensemble recommendation
            ensemble_recommendation = {
                'models': [model1, model2],
                'ensemble_method': 'weighted_average',
                'weights': [0.5, 0.5]  # Default to equal weights
            }
            
            # Customize weights based on model scores
            if model1 in model_scores and model2 in model_scores:
                total = model_scores[model1] + model_scores[model2]
                if total > 0:
                    ensemble_recommendation['weights'] = [
                        round(model_scores[model1] / total, 2),
                        round(model_scores[model2] / total, 2)
                    ]
            
            # Create explanation
            explanation = f"An ensemble approach is recommended combining {model1} and {model2}. {pair_reason}.\n\n"
            explanation += "Reasons for recommending an ensemble:\n"
            for reason in reasons:
                explanation += f"- {reason}\n"
            
            explanation += f"\nSuggested implementation: Use a weighted average ensemble with weights {ensemble_recommendation['weights'][0]:.2f} for {model1} and {ensemble_recommendation['weights'][1]:.2f} for {model2}."
            explanation += "\n\nAlternative ensemble methods to consider:"
            explanation += "\n- Stacking: Train a meta-model that learns how to best combine the base models"
            explanation += "\n- Boosting: Sequential training where each model corrects errors of previous models"
            explanation += "\n- Simple average: Equal weighting if uncertain about optimal weights"
    
    return ensemble_recommendation, explanation

def implement_ensemble_forecast(models, X_train, y_train, X_test, weights=None):
    """
    Implements a basic ensemble forecast using the specified models and weights.
    
    Parameters:
    -----------
    models : list
        List of fitted model objects
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    X_test : array-like
        Test features
    weights : list, optional
        List of weights for each model. If None, equal weights are used.
        
    Returns:
    --------
    array-like : Ensemble predictions
    """
    if weights is None:
        weights = [1/len(models)] * len(models)
    
    # Ensure weights sum to 1
    weights = np.array(weights) / sum(weights)
    
    predictions = []
    for i, model in enumerate(models):
        # This is a simplified implementation - actual implementation would depend on model types
        preds = model.predict(X_test)
        predictions.append(preds * weights[i])
    
    # Sum weighted predictions
    ensemble_preds = sum(predictions)
    return ensemble_preds

def plot_analysis(df, target_col=None, datetime_col=None):
    """
    Plot visualizations to help understand the time series characteristics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The timeseries dataset
    target_col : str, optional
        The target column to analyze
    datetime_col : str, optional
        The column containing datetime information
        
    Returns:
    --------
    None (displays plots)
    """
    # Preprocessing similar to analyze_timeseries
    if isinstance(df, pd.Series):
        series = df
        df = pd.DataFrame(df)
        target_col = df.columns[0]
    elif target_col is None and df.shape[1] == 1:
        target_col = df.columns[0]
        series = df[target_col]
    elif target_col is not None:
        series = df[target_col]
    else:
        raise ValueError("For multivariate data, please specify the target column.")
    
    # Ensure datetime index
    if datetime_col is not None:
        df = df.set_index(datetime_col)
    
    # Check if index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            raise ValueError("Index must be convertible to datetime. Please provide a datetime_col.")
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Original Time Series
    plt.subplot(4, 1, 1)
    plt.plot(series)
    plt.title('Original Time Series')
    plt.grid(True)
    
    # Plot 2: ACF
    plt.subplot(4, 1, 2)
    try:
        acf_values = acf(series.dropna(), nlags=min(40, len(series)//4))
        plt.bar(range(len(acf_values)), acf_values)
        plt.axhline(y=0, linestyle='-', color='black')
        plt.axhline(y=1.96/np.sqrt(len(series)), linestyle='--', color='red')
        plt.axhline(y=-1.96/np.sqrt(len(series)), linestyle='--', color='red')
        plt.title('Autocorrelation Function')
    except:
        plt.text(0.5, 0.5, 'Could not calculate ACF', ha='center')
    
    # Plot 3: Seasonal Decomposition if possible
    plt.subplot(4, 1, 3)
    try:
        # Try to infer frequency if not explicitly provided
        if df.index.freq is None:
            df = df.asfreq(pd.infer_freq(df.index))
            
        decomposition = seasonal_decompose(series.dropna(), model='additive')
        plt.plot(decomposition.trend, label='Trend')
        plt.plot(decomposition.seasonal, label='Seasonality')
        plt.legend()
        plt.title('Trend and Seasonality Components')
        plt.grid(True)
    except:
        plt.text(0.5, 0.5, 'Could not perform seasonal decomposition', ha='center')
    
    # Plot 4: Residuals from decomposition
    plt.subplot(4, 1, 4)
    try:
        if df.index.freq is None:
            df = df.asfreq(pd.infer_freq(df.index))
            
        decomposition = seasonal_decompose(series.dropna(), model='additive')
        plt.plot(decomposition.resid, label='Residuals')
        plt.axhline(y=0, linestyle='-', color='black')
        plt.legend()
        plt.title('Residuals (after removing trend and seasonality)')
        plt.grid(True)
    except:
        plt.text(0.5, 0.5, 'Could not calculate residuals', ha='center')
    
    plt.tight_layout()
    plt.show()

def model_selection_pipeline(dataset_path, target_col=None, experiment_name="Model_Selection_Experiment"):
    """
    Main function to launch the Model Selection and Analysis Pipeline with MLflow tracking.
    
    Args:
        dataset_path (str): Path to the CSV dataset
        target_col (str): Target column to forecast (optional)
        experiment_name (str): MLflow experiment name
    """
    # 1. Charger les données
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    target_col = df.columns[-1] if target_col is None else target_col

    # 2. Démarrer l'expérience MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"Model_Selection_for_{target_col}"):

        # 3. Analyse du Time Series
        result = analyze_timeseries(df, target_col=target_col)

        # 4. Logging dans MLflow
        analysis = result.get('analysis', {})

        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("dataset_length", analysis.get('length'))
        mlflow.log_param("variables", analysis.get('variables'))
        mlflow.log_param("stationary", analysis.get('stationary'))
        mlflow.log_param("has_trend", analysis.get('has_trend'))
        mlflow.log_param("has_seasonality", analysis.get('has_seasonality'))
        mlflow.log_param("complex_patterns", analysis.get('complex_patterns'))
        mlflow.log_param("size", analysis.get('size'))
        mlflow.log_param("frequency", str(analysis.get('frequency')))

        # 5. Résultats de la recommandation
        if result['recommended_approach'] == 'ensemble':
            models = ' + '.join(result['ensemble_details']['models'])
            mlflow.log_param("recommended_approach", "ensemble")
            mlflow.log_param("ensemble_models", models)
            mlflow.log_param("ensemble_method", result['ensemble_details']['ensemble_method'])
            print(f"Recommended Approach: Ensemble of {models}")
            print(f"\nExplanation:\n{result['explanation']}")
        else:
            mlflow.log_param("recommended_approach", "single_model")
            mlflow.log_param("recommended_model", result['recommended_model'])
            print(f"Recommended Model: {result['recommended_model']}")
            print(f"\nExplanation:\n{result['explanation']}")

        # 6. Génération et enregistrement des graphiques
        plot_analysis(df[target_col])
        
        # 7. Save recommended model(s) into a text file for Prefect
        with open("recommended_model.txt", "w") as f:
            if result['recommended_approach'] == 'ensemble':
                models = result['ensemble_details']['models']
                f.write(f"ensemble:{'+'.join(models)}")
            else:
                f.write(result['recommended_model'])

        
        # Optionally, you can save the plots and log them as MLflow artifacts here if you want (easy to add)

    print("✅ Model selection completed and logged in MLflow.")

# Example usage:
# if __name__ == "__main__":
#     model_selection_pipeline("../datasets/energydata_complete.csv")
