import time

import numpy as np


def matching_latency(
    method, dataset, warmup_iter: int = 3, num_samples: int = 10
) -> float:
    method.compute_features(dataset)
    dl = dataset.dataloader(batch_size=1, num_workers=0, transform=method.transform)
    for batch, idx in dl:
        break

    method.eval()
    batch = batch.to(next(method.parameters()).device)
    query = method(batch)
    samples = []
    for _ in range(warmup_iter):
        method.match(query, topk=1)

    for _ in range(num_samples):
        start = time.time()
        method.match(query, topk=1)
        end = time.time()
        samples.append((end - start) * 1000)  # Convert to milliseconds
    return np.mean(samples)
