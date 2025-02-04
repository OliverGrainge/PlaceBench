import numpy as np 

def ratk(method, dataset, topk=1) -> float: 
    method.load_database(dataset, recompute=True) 
    matches = method.recognize(dataset, topk=topk, recompute=True) 
   
    ground_truth = dataset.ground_truth() 

    correct_at_k = np.zeros(1, dtype=np.float32)
    for q_idx, match in enumerate(matches):
        # Use vectorized operations for better performance
        found_at = np.where(np.in1d(match, ground_truth[q_idx]))[0]
        if len(found_at) > 0:
            first_match = found_at[0]
            correct_at_k += first_match < np.array(topk)
    return correct_at_k / len(matches)



# Calculate accuracy for each k value
