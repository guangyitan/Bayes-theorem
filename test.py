def calculate_error(feature_values, labels):
    """
    Calculate the error rate for a given feature and label pair.
    
    Parameters:
        feature_values (list): A list of feature values.
        labels (list): A list of corresponding class labels.
        
    Returns:
        The error rate for this feature.
    """
    # Count the frequency of each class label for each feature value
    frequency_table = {}
    for feature_value, label in zip(feature_values, labels):
        if feature_value not in frequency_table:
            frequency_table[feature_value] = {}
        if label not in frequency_table[feature_value]:
            frequency_table[feature_value][label] = 0
        frequency_table[feature_value][label] += 1
    
    # Calculate the total number of samples and the number of errors
    total_samples = len(feature_values)
    total_errors = 0
    for feature_value, label_counts in frequency_table.items():
        most_common_label = max(label_counts, key=label_counts.get)
        error_count = sum(label_counts.values()) - label_counts[most_common_label]
        total_errors += error_count
    
    # Return the error rate
    return total_errors / total_samples


def OneR_classification(train_features, train_labels):
    """
    Implements the OneR classification algorithm, which selects a single feature to make predictions.
    
    Parameters:
        train_features (list of lists): A list of training samples, where each sample is a list of feature values.
        train_labels (list): A list of labels from the training set.
        
    Returns:
        A dictionary containing the selected feature and the corresponding error rate.
    """
    # Find the best feature by calculating the error rate for each feature
    best_feature = None
    best_error_rate = float('inf')
    for feature_index in range(len(train_features[0])):
        feature_values = [sample[feature_index] for sample in train_features]
        error_rate = calculate_error(feature_values, train_labels)
        if error_rate < best_error_rate:
            best_feature = feature_index
            best_error_rate = error_rate
    
    # Create a dictionary containing the best feature and the error rate
    result = {}
    result['best_feature'] = best_feature
    result['error_rate'] = best_error_rate
    
    return result
