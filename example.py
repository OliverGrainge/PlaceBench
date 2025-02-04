from datasets import Pitts30k
from methods import EigenPlaces 
from metrics import ratk, extraction_latency, matching_latency, model_memory, database_memory
import config 


dataset = Pitts30k(config.Pitts30k_root)
method = EigenPlaces()

#print(ratk(method, dataset, topks=[1, 5]))
#print(extraction_latency(method, dataset))
#print(matching_latency(method, dataset))
#print(model_memory(method, dataset))
print(database_memory(method, dataset))