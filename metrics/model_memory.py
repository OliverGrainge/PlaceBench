import numpy as np


def model_memory(method, dataset=None) -> float:
    sd = method.state_dict()

    # Calculate memory based on parameter dtype
    total_bytes = sum(p.numel() * p.element_size() for p in sd.values())
    return total_bytes / (1024**2)  # Convert bytes to MB
