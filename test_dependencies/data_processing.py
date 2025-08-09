"""
Data processing utilities module
Contains core data transformation functions
"""

def normalize_data(data_list):
    """Normalize a list of numbers to 0-1 range"""
    if not data_list:
        return []
    
    min_val = min(data_list)
    max_val = max(data_list)
    
    if min_val == max_val:
        return [0.5] * len(data_list)
    
    normalized = []
    for value in data_list:
        normalized_value = (value - min_val) / (max_val - min_val)
        normalized.append(normalized_value)
    
    return normalized


def filter_outliers(data_list, threshold=2.0):
    """Filter out outliers using standard deviation method"""
    if len(data_list) < 2:
        return data_list
    
    mean_val = calculate_mean(data_list)
    std_dev = calculate_standard_deviation(data_list, mean_val)
    
    filtered_data = []
    for value in data_list:
        if abs(value - mean_val) <= threshold * std_dev:
            filtered_data.append(value)
    
    return filtered_data


def calculate_mean(numbers):
    """Calculate the arithmetic mean of a list of numbers"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)


def calculate_standard_deviation(numbers, mean_val=None):
    """Calculate standard deviation of a list of numbers"""
    if not numbers:
        return 0
    
    if mean_val is None:
        mean_val = calculate_mean(numbers)
    
    variance_sum = 0
    for num in numbers:
        variance_sum += (num - mean_val) ** 2
    
    variance = variance_sum / len(numbers)
    return variance ** 0.5


def transform_data(raw_data, remove_outliers=True, normalize=True):
    """Complete data transformation pipeline"""
    # Start with raw data
    processed_data = list(raw_data)
    
    # Step 1: Filter outliers if requested
    if remove_outliers:
        processed_data = filter_outliers(processed_data)
        print(f"Filtered data: {len(processed_data)} items remaining")
    
    # Step 2: Normalize data if requested
    if normalize:
        processed_data = normalize_data(processed_data)
        print("Data normalized to 0-1 range")
    
    return processed_data


def validate_data_quality(data_list):
    """Validate the quality of processed data"""
    if not data_list:
        return {"valid": False, "errors": ["Empty dataset"]}
    
    mean_val = calculate_mean(data_list)
    std_dev = calculate_standard_deviation(data_list, mean_val)
    
    quality_report = {
        "valid": True,
        "size": len(data_list),
        "mean": mean_val,
        "std_dev": std_dev,
        "errors": []
    }
    
    # Check for potential issues
    if std_dev == 0:
        quality_report["errors"].append("Zero variance - all values identical")
    
    if any(x < 0 for x in data_list):
        quality_report["errors"].append("Contains negative values")
    
    return quality_report
