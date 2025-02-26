from datasets import Essex3in1
from methods import EigenPlacesD2048
from metrics import ratk, model_memory, matching_latency, extraction_latency_cpu, database_memory
import config 


dataset = Essex3in1(root=config.Essex3in1_root)
method = EigenPlacesD2048()

method.compute_features(dataset, num_workers=0, batch_size=12) 

rk = ratk(method, dataset, topks=[1])
mlat = matching_latency(method, dataset)
elat = extraction_latency_cpu = extraction_latency_cpu(method, dataset)
db_mem = database_memory = database_memory(method, dataset)
mod_mem = model_memory = model_memory(method)


print(f"================ {method.name} metrics ==================")
print(f"R@1: {rk[0]:.3f}")
print(f"Matching latency: {mlat:.3f} ms")
print(f"Feature extraction latency: {elat:.3f} ms")
print(f"Database memory: {db_mem:.2f} MB")
print(f"Model memory: {mod_mem:.2f} MB")