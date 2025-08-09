"""
Machine Learning utilities module
Contains model training and evaluation functions
"""

from data_processing import transform_data, validate_data_quality, calculate_mean
import math


class SimpleLinearRegression:
    """Simple linear regression model"""
    
    def __init__(self):
        self.slope = 0
        self.intercept = 0
        self.is_trained = False
    
    def train(self, x_data, y_data):
        """Train the linear regression model"""
        if len(x_data) != len(y_data):
            raise ValueError("X and Y data must have same length")
        
        # Prepare training data
        clean_x = prepare_training_data(x_data)
        clean_y = prepare_training_data(y_data)
        
        # Calculate slope and intercept
        self.slope, self.intercept = calculate_regression_coefficients(clean_x, clean_y)
        self.is_trained = True
        
        return self
    
    def predict(self, x_values):
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        predictions = []
        for x in x_values:
            prediction = self.slope * x + self.intercept
            predictions.append(prediction)
        
        return predictions


def prepare_training_data(raw_data):
    """Prepare raw data for machine learning training"""
    # Use data processing utilities to clean the data
    cleaned_data = transform_data(raw_data, remove_outliers=True, normalize=False)
    
    # Additional ML-specific preprocessing
    if not cleaned_data:
        raise ValueError("No valid data remaining after preprocessing")
    
    # Ensure data quality meets ML requirements
    quality_report = validate_data_quality(cleaned_data)
    if not quality_report["valid"]:
        raise ValueError(f"Data quality issues: {quality_report['errors']}")
    
    return cleaned_data


def calculate_regression_coefficients(x_data, y_data):
    """Calculate slope and intercept for linear regression"""
    n = len(x_data)
    if n < 2:
        raise ValueError("Need at least 2 data points for regression")
    
    # Calculate means
    x_mean = calculate_mean(x_data)
    y_mean = calculate_mean(y_data)
    
    # Calculate slope
    numerator = 0
    denominator = 0
    
    for i in range(n):
        x_diff = x_data[i] - x_mean
        y_diff = y_data[i] - y_mean
        numerator += x_diff * y_diff
        denominator += x_diff * x_diff
    
    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator
    
    # Calculate intercept
    intercept = y_mean - slope * x_mean
    
    return slope, intercept


def evaluate_model_performance(model, test_x, test_y):
    """Evaluate model performance using various metrics"""
    predictions = model.predict(test_x)
    
    # Calculate evaluation metrics
    mse = calculate_mean_squared_error(test_y, predictions)
    rmse = math.sqrt(mse)
    mae = calculate_mean_absolute_error(test_y, predictions)
    r2 = calculate_r_squared(test_y, predictions)
    
    return {
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "mean_absolute_error": mae,
        "r_squared": r2,
        "sample_size": len(test_y)
    }


def calculate_mean_squared_error(actual, predicted):
    """Calculate Mean Squared Error"""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted must have same length")
    
    squared_errors = []
    for i in range(len(actual)):
        error = actual[i] - predicted[i]
        squared_errors.append(error * error)
    
    return calculate_mean(squared_errors)


def calculate_mean_absolute_error(actual, predicted):
    """Calculate Mean Absolute Error"""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted must have same length")
    
    absolute_errors = []
    for i in range(len(actual)):
        error = abs(actual[i] - predicted[i])
        absolute_errors.append(error)
    
    return calculate_mean(absolute_errors)


def calculate_r_squared(actual, predicted):
    """Calculate R-squared coefficient of determination"""
    actual_mean = calculate_mean(actual)
    
    total_sum_squares = 0
    residual_sum_squares = 0
    
    for i in range(len(actual)):
        total_sum_squares += (actual[i] - actual_mean) ** 2
        residual_sum_squares += (actual[i] - predicted[i]) ** 2
    
    if total_sum_squares == 0:
        return 1.0
    
    return 1 - (residual_sum_squares / total_sum_squares)


def cross_validate_model(x_data, y_data, k_folds=5):
    """Perform k-fold cross validation"""
    if len(x_data) < k_folds:
        raise ValueError("Not enough data for k-fold validation")
    
    # Prepare data
    clean_x = prepare_training_data(x_data)
    clean_y = prepare_training_data(y_data)
    
    fold_size = len(clean_x) // k_folds
    performance_scores = []
    
    for fold in range(k_folds):
        # Create train/test split for this fold
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < k_folds - 1 else len(clean_x)
        
        test_x = clean_x[start_idx:end_idx]
        test_y = clean_y[start_idx:end_idx]
        train_x = clean_x[:start_idx] + clean_x[end_idx:]
        train_y = clean_y[:start_idx] + clean_y[end_idx:]
        
        # Train model on fold
        model = SimpleLinearRegression()
        model.train(train_x, train_y)
        
        # Evaluate on test set
        fold_performance = evaluate_model_performance(model, test_x, test_y)
        performance_scores.append(fold_performance)
    
    return aggregate_cross_validation_results(performance_scores)


def aggregate_cross_validation_results(performance_scores):
    """Aggregate results from cross-validation folds"""
    metrics = ["mean_squared_error", "root_mean_squared_error", "mean_absolute_error", "r_squared"]
    
    aggregated = {}
    for metric in metrics:
        values = [score[metric] for score in performance_scores]
        aggregated[f"{metric}_mean"] = calculate_mean(values)
        aggregated[f"{metric}_std"] = calculate_standard_deviation(values)
    
    aggregated["num_folds"] = len(performance_scores)
    return aggregated


# Import calculate_standard_deviation from data_processing
from data_processing import calculate_standard_deviation
