import time
import numpy as np 

def extraction_latency(method, dataset, warmup_iter: int=10, num_samples: int=100, device: str="cuda") -> float:
    dl = dataset.dataloader(batch_size=1, num_workers=0, transform=method.transform)
    for batch, idx in dl: 
        break
    
    method = method.to(device)
    method.eval()
    batch = batch.to(device)
    # Warmup phase
    for _ in range(warmup_iter):
        method(batch)
    # Reset dataloader for actual measurement
    samples = []
    
    for _ in range(num_samples): 
        start = time.time()
        method(batch)
        end = time.time()
        samples.append((end - start) * 1000)  # Convert to milliseconds
    return np.mean(samples)