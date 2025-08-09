"""
Data visualization and reporting module
Contains functions for creating reports and visualizations
"""

from data_processing import validate_data_quality, calculate_mean, calculate_standard_deviation
from ml_utils import evaluate_model_performance, SimpleLinearRegression
import json


class DataReport:
    """Class for generating comprehensive data analysis reports"""
    
    def __init__(self, dataset_name="Unknown Dataset"):
        self.dataset_name = dataset_name
        self.sections = []
        self.metadata = {
            "created_by": "Data Analysis Pipeline",
            "dataset": dataset_name
        }
    
    def add_section(self, title, content):
        """Add a section to the report"""
        section = {
            "title": title,
            "content": content,
            "order": len(self.sections)
        }
        self.sections.append(section)
    
    def generate_summary_statistics(self, data):
        """Generate summary statistics for a dataset"""
        stats = calculate_descriptive_statistics(data)
        quality = validate_data_quality(data)
        
        summary = {
            "basic_stats": stats,
            "quality_assessment": quality,
            "recommendations": generate_recommendations(stats, quality)
        }
        
        self.add_section("Summary Statistics", summary)
        return summary
    
    def generate_model_report(self, model, test_x, test_y):
        """Generate a comprehensive model evaluation report"""
        performance = evaluate_model_performance(model, test_x, test_y)
        interpretation = interpret_model_performance(performance)
        
        model_report = {
            "model_type": type(model).__name__,
            "performance_metrics": performance,
            "interpretation": interpretation,
            "training_data_size": len(test_x)
        }
        
        self.add_section("Model Performance", model_report)
        return model_report
    
    def export_report(self, format_type="json"):
        """Export the complete report"""
        complete_report = {
            "metadata": self.metadata,
            "sections": self.sections,
            "total_sections": len(self.sections)
        }
        
        if format_type == "json":
            return json.dumps(complete_report, indent=2)
        elif format_type == "summary":
            return generate_text_summary(complete_report)
        else:
            return str(complete_report)


def calculate_descriptive_statistics(data):
    """Calculate comprehensive descriptive statistics"""
    if not data:
        return {"error": "No data provided"}
    
    # Basic statistics
    mean_val = calculate_mean(data)
    std_dev = calculate_standard_deviation(data, mean_val)
    
    # Additional statistics
    sorted_data = sorted(data)
    min_val = sorted_data[0]
    max_val = sorted_data[-1]
    median_val = calculate_median(sorted_data)
    
    # Quartiles
    q1 = calculate_quartile(sorted_data, 0.25)
    q3 = calculate_quartile(sorted_data, 0.75)
    iqr = q3 - q1
    
    return {
        "count": len(data),
        "mean": mean_val,
        "median": median_val,
        "std_deviation": std_dev,
        "min": min_val,
        "max": max_val,
        "range": max_val - min_val,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "coefficient_of_variation": std_dev / mean_val if mean_val != 0 else 0
    }


def calculate_median(sorted_data):
    """Calculate median of sorted data"""
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        middle1 = sorted_data[n // 2 - 1]
        middle2 = sorted_data[n // 2]
        return (middle1 + middle2) / 2


def calculate_quartile(sorted_data, percentile):
    """Calculate quartile/percentile of sorted data"""
    n = len(sorted_data)
    index = percentile * (n - 1)
    
    if index == int(index):
        return sorted_data[int(index)]
    else:
        lower_idx = int(index)
        upper_idx = lower_idx + 1
        weight = index - lower_idx
        return sorted_data[lower_idx] * (1 - weight) + sorted_data[upper_idx] * weight


def generate_recommendations(stats, quality):
    """Generate data analysis recommendations based on statistics and quality"""
    recommendations = []
    
    # Check for data quality issues
    if quality.get("errors"):
        recommendations.append("Address data quality issues: " + ", ".join(quality["errors"]))
    
    # Check sample size
    if stats.get("count", 0) < 30:
        recommendations.append("Consider collecting more data - sample size is small for reliable analysis")
    
    # Check for high variability
    cv = stats.get("coefficient_of_variation", 0)
    if cv > 1.0:
        recommendations.append("High variability detected - consider investigating outliers or data collection process")
    
    # Check for skewness (rough approximation)
    mean_val = stats.get("mean", 0)
    median_val = stats.get("median", 0)
    if abs(mean_val - median_val) > 0.1 * stats.get("std_deviation", 1):
        recommendations.append("Data appears skewed - consider data transformation or non-parametric methods")
    
    if not recommendations:
        recommendations.append("Data appears to be of good quality for analysis")
    
    return recommendations


def interpret_model_performance(performance_metrics):
    """Interpret model performance metrics and provide insights"""
    interpretations = []
    
    # Interpret R-squared
    r2 = performance_metrics.get("r_squared", 0)
    if r2 > 0.9:
        interpretations.append("Excellent model fit (R² > 0.9)")
    elif r2 > 0.7:
        interpretations.append("Good model fit (R² > 0.7)")
    elif r2 > 0.5:
        interpretations.append("Moderate model fit (R² > 0.5)")
    else:
        interpretations.append("Poor model fit (R² ≤ 0.5) - consider different model or feature engineering")
    
    # Interpret RMSE relative to data range
    rmse = performance_metrics.get("root_mean_squared_error", 0)
    interpretations.append(f"Average prediction error (RMSE): {rmse:.4f}")
    
    # Sample size assessment
    sample_size = performance_metrics.get("sample_size", 0)
    if sample_size < 20:
        interpretations.append("Small test set - results may not be reliable")
    
    return interpretations


def generate_text_summary(report_data):
    """Generate a human-readable text summary of the report"""
    summary_lines = []
    summary_lines.append(f"=== Data Analysis Report ===")
    summary_lines.append(f"Dataset: {report_data['metadata']['dataset']}")
    summary_lines.append(f"Total Sections: {report_data['total_sections']}")
    summary_lines.append("")
    
    for section in report_data['sections']:
        summary_lines.append(f"Section: {section['title']}")
        
        if section['title'] == "Summary Statistics":
            content = section['content']
            basic_stats = content.get('basic_stats', {})
            summary_lines.append(f"  Sample Size: {basic_stats.get('count', 'N/A')}")
            summary_lines.append(f"  Mean: {basic_stats.get('mean', 'N/A'):.4f}")
            summary_lines.append(f"  Std Dev: {basic_stats.get('std_deviation', 'N/A'):.4f}")
            
        elif section['title'] == "Model Performance":
            content = section['content']
            performance = content.get('performance_metrics', {})
            summary_lines.append(f"  R-squared: {performance.get('r_squared', 'N/A'):.4f}")
            summary_lines.append(f"  RMSE: {performance.get('root_mean_squared_error', 'N/A'):.4f}")
        
        summary_lines.append("")
    
    return "\n".join(summary_lines)


def create_comprehensive_analysis_pipeline(raw_x_data, raw_y_data, dataset_name="Analysis"):
    """
    Complete analysis pipeline that demonstrates complex function dependencies
    This function uses almost every other function in the codebase
    """
    # Initialize report
    report = DataReport(dataset_name)
    
    # Step 1: Generate summary statistics for input data
    x_summary = report.generate_summary_statistics(raw_x_data)
    y_summary = report.generate_summary_statistics(raw_y_data)
    
    # Step 2: Create and train model
    model = SimpleLinearRegression()
    
    try:
        # Prepare and train model
        model.train(raw_x_data, raw_y_data)
        
        # Step 3: Evaluate model performance
        model_report = report.generate_model_report(model, raw_x_data, raw_y_data)
        
        # Step 4: Generate final recommendations
        final_recommendations = combine_recommendations(
            x_summary.get('recommendations', []),
            y_summary.get('recommendations', []),
            model_report.get('interpretation', [])
        )
        
        report.add_section("Final Recommendations", final_recommendations)
        
        # Export complete report
        final_report = report.export_report("summary")
        
        return {
            "success": True,
            "model": model,
            "report": final_report,
            "performance": model_report['performance_metrics']
        }
        
    except Exception as e:
        error_report = {
            "success": False,
            "error": str(e),
            "partial_analysis": report.export_report("json")
        }
        return error_report


def combine_recommendations(x_recommendations, y_recommendations, model_recommendations):
    """Combine recommendations from different analysis stages"""
    combined = []
    
    # Add data-specific recommendations
    if x_recommendations:
        combined.append("X-data recommendations: " + "; ".join(x_recommendations))
    
    if y_recommendations:
        combined.append("Y-data recommendations: " + "; ".join(y_recommendations))
    
    # Add model-specific recommendations
    if model_recommendations:
        combined.append("Model recommendations: " + "; ".join(model_recommendations))
    
    # Add overall recommendation
    combined.append("Consider validating results with domain experts before making decisions")
    
    return combined
