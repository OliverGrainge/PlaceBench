# Visual Place Recognition Benchmarking

A comprehensive benchmarking toolkit for evaluating Visual Place Recognition (VPR) models and methods.

## Overview

This repository provides tools and metrics to evaluate various Visual Place Recognition models across different performance aspects, including:
- Recognition accuracy
- Computational efficiency
- Memory requirements
- Processing latency

## Features

- **Multiple Metrics Support:**
  - R@k (Recall at k)
  - Matching latency
  - Feature extraction latency
  - Database memory usage
  - Model memory footprint

- **Dataset Integration:**
  - Essex3in1 dataset support
  - Extensible for additional datasets

- **Method Implementation:**
  - EigenPlaces implementation
  - Easy integration of new VPR methods

## Installation
```bash
git clone https://github.com/OliverGrainge/PlaceBench.git
cd PlaceBench
pip install -r requirements.txt
```

## Usage

Basic example of running a benchmark:

```python

from datasets import Essex3in1
from methods import EigenPlacesD2048
from metrics import ratk, model_memory, matching_latency, extraction_latency_cpu, database_memory
import config

# pick a dataset from datasets/
dataset = Essex3in1(root=config.Essex3in1_root)

# pick a method from methods/
method = EigenPlacesD2048()


# compute features for the dataset 
method.compute_features(dataset, num_workers=0, batch_size=12)

# compute metrics
rk = ratk(method, dataset, topks=[1])
mlat = matching_latency(method, dataset)
elat = extraction_latency_cpu(method, dataset)
db_mem = database_memory(method, dataset)
mod_mem = model_memory(method)

print(f"================ {method.name} metrics ==================")
print(f"R@1: {rk[0]:.3f}")
print(f"Matching latency: {mlat:.3f} ms")
print(f"Feature extraction latency: {elat:.3f} ms")
print(f"Database memory: {db_mem:.2f} MB")
print(f"Model memory: {mod_mem:.2f} MB")
```

## Metrics

- **R@k**: Recall at k, measuring recognition accuracy
- **Matching Latency**: Time required for place matching (in milliseconds)
- **Feature Extraction Latency**: Time required to extract features from images (in milliseconds)
- **Database Memory**: Memory required to store the feature database (in MB)
- **Model Memory**: Memory footprint of the model (in MB)

## Adding New Methods

To add a new VPR method, create a new class in the `methods` directory implementing the required interface.

## Supported Datasets

- Essex3in1
- (Add other supported datasets here)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Citation

If you use this benchmark in your research, please cite: