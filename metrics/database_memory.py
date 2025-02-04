import numpy as np 

def database_memory(method, dataset) -> float:
    features = method.compute_features(dataset)
    # Sum memory usage of all NumPy arrays in the database dictionary (converted to MB)
    return sum(arr.nbytes for arr in features["database"].values()) / (1024 ** 2)