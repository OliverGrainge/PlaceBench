import numpy as np 
from typing import List


def ratk(method, dataset, topks=[1, 5]) -> np.ndarray: 
    features = method.compute_features(dataset) 
    matches = method.match(features["query"], topk=max(topks))
    ground_truth = dataset.ground_truth() 

    correct_at_k = np.zeros(len(topks), dtype=np.float32)
    for q_idx, match in enumerate(matches):
        # Use vectorized operations for better performance
        found_at = np.where(np.in1d(match, ground_truth[q_idx]))[0]
        if len(found_at) > 0:
            first_match = found_at[0]
            correct_at_k += first_match < np.array(topks)
    return (correct_at_k / len(matches)) * 100



# Calculate accuracy for each k value
