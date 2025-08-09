"""
Main analysis script demonstrating complex function dependencies
This serves as the entry point for the data analysis pipeline
"""

from reporting import create_comprehensive_analysis_pipeline, DataReport
from ml_utils import cross_validate_model, SimpleLinearRegression
from data_processing import transform_data, validate_data_quality
import random


def generate_sample_data(n_samples=100, noise_level=0.1):
    """Generate sample data for testing the analysis pipeline"""
    # Generate X data with some pattern
    x_data = []
    for i in range(n_samples):
        x_val = i / 10.0 + random.uniform(-0.5, 0.5)
        x_data.append(x_val)
    
    # Generate Y data correlated with X plus noise
    y_data = []
    for x in x_data:
        y_val = 2.5 * x + 1.0 + random.uniform(-noise_level, noise_level) * 10
        y_data.append(y_val)
    
    return x_data, y_data


def run_basic_analysis(x_data, y_data):
    """Run basic analysis without full pipeline"""
    print("=== Basic Analysis ===")
    
    # Data quality check
    x_quality = validate_data_quality(x_data)
    y_quality = validate_data_quality(y_data)
    
    print(f"X data quality: {x_quality['valid']}, Size: {x_quality['size']}")
    print(f"Y data quality: {y_quality['valid']}, Size: {y_quality['size']}")
    
    # Simple model training
    model = SimpleLinearRegression()
    model.train(x_data, y_data)
    
    print(f"Model trained - Slope: {model.slope:.4f}, Intercept: {model.intercept:.4f}")
    
    # Test predictions
    test_x = [1.0, 2.0, 3.0]
    predictions = model.predict(test_x)
    print(f"Test predictions for {test_x}: {[f'{p:.4f}' for p in predictions]}")
    
    return model


def run_advanced_analysis(x_data, y_data):
    """Run advanced analysis with cross-validation"""
    print("\n=== Advanced Analysis ===")
    
    try:
        # Cross-validation analysis
        cv_results = cross_validate_model(x_data, y_data, k_folds=3)
        
        print("Cross-validation results:")
        for metric, value in cv_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.6f}")
            else:
                print(f"  {metric}: {value}")
                
        return cv_results
        
    except Exception as e:
        print(f"Advanced analysis failed: {e}")
        return None


def run_comprehensive_pipeline(x_data, y_data):
    """
    Run the complete comprehensive analysis pipeline
    This function has the most complex dependency chain
    """
    print("\n=== Comprehensive Pipeline Analysis ===")
    
    try:
        # This function calls almost every other function in the codebase
        result = create_comprehensive_analysis_pipeline(
            x_data, y_data, 
            dataset_name="Sample Dataset Analysis"
        )
        
        if result['success']:
            print("Pipeline completed successfully!")
            print("\nFinal Report:")
            print(result['report'])
            
            # Additional model testing
            model = result['model']
            test_predictions = model.predict([0, 5, 10])
            print(f"\nTest predictions: {[f'{p:.4f}' for p in test_predictions]}")
            
            return result
        else:
            print(f"Pipeline failed: {result['error']}")
            return result
            
    except Exception as e:
        print(f"Comprehensive pipeline error: {e}")
        return {"success": False, "error": str(e)}


def compare_analysis_approaches(x_data, y_data):
    """Compare different analysis approaches"""
    print("\n=== Comparing Analysis Approaches ===")
    
    # Approach 1: Raw data
    raw_model = SimpleLinearRegression()
    raw_model.train(x_data, y_data)
    raw_predictions = raw_model.predict([1, 2, 3])
    
    # Approach 2: Transformed data
    transformed_x = transform_data(x_data, remove_outliers=True, normalize=True)
    transformed_y = transform_data(y_data, remove_outliers=True, normalize=True)
    
    if len(transformed_x) > 1 and len(transformed_y) > 1:
        transformed_model = SimpleLinearRegression()
        transformed_model.train(transformed_x, transformed_y)
        transformed_predictions = transformed_model.predict([0.1, 0.5, 0.9])
        
        print("Raw data model predictions:", [f'{p:.4f}' for p in raw_predictions])
        print("Transformed data model predictions:", [f'{p:.4f}' for p in transformed_predictions])
    else:
        print("Not enough data after transformation for comparison")


def main():
    """
    Main function that orchestrates the entire analysis
    This function demonstrates the most complex dependency chain
    """
    print("Starting Complex Data Analysis Pipeline")
    print("=" * 50)
    
    # Generate test data
    print("Generating sample data...")
    x_data, y_data = generate_sample_data(n_samples=50, noise_level=0.2)
    
    # Run different analysis approaches
    basic_model = run_basic_analysis(x_data, y_data)
    
    advanced_results = run_advanced_analysis(x_data, y_data)
    
    comprehensive_results = run_comprehensive_pipeline(x_data, y_data)
    
    compare_analysis_approaches(x_data, y_data)
    
    # Final summary
    print("\n" + "=" * 50)
    print("Analysis Complete!")
    
    if comprehensive_results and comprehensive_results.get('success'):
        performance = comprehensive_results['performance']
        print(f"Final Model Performance:")
        print(f"  R²: {performance.get('r_squared', 'N/A'):.4f}")
        print(f"  RMSE: {performance.get('root_mean_squared_error', 'N/A'):.4f}")
    
    return {
        "basic_model": basic_model,
        "advanced_results": advanced_results,
        "comprehensive_results": comprehensive_results
    }


# Entry point for testing
if __name__ == "__main__":
    results = main()
