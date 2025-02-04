from abc import ABC, abstractmethod
import torch 
import torch.nn as nn 
from typing import Union, List
from PIL import Image
import numpy as np 
from torchvision import transforms
import faiss
import os 
import pickle 
from tqdm import tqdm 



class SingleStageMethod(nn.Module): 
    def __init__(self, name: str, model: nn.Module, transform: transforms.Compose, descriptor_dim: int, search_dist: str='cosine'): 
        super().__init__()
        self.name = name 
        self.model = self._freeze_model(model)
        self.transform = transform
        self.descriptor_dim = descriptor_dim 
        self.search_dist = search_dist
        self.model.eval()

    def forward(self, input: Union[Image.Image, torch.Tensor]) -> dict: 
        if isinstance(input, Image.Image): 
            image = self.transform(image)[None, ...]
        return self.model(input)
    
    def compute_features(self, dataset, batch_size: int=32, num_workers: int=4, recompute: bool=False, device: Union[str, None]=None) -> dict: 
        if not recompute: 
            feature_dict = self._load_features(dataset.name)
            if feature_dict is not None: 
                self.index = self._setup_index(feature_dict["database"]["global_desc"])
                return feature_dict
                
        if device is None: 
            device = self._detect_device() 

        self.model = self.model.to(device)
        self.model.eval()

        dl = dataset.dataloader(batch_size=batch_size, num_workers=num_workers, transform=self.transform)
        all_desc = np.zeros((len(dataset), self.descriptor_dim))
        for batch in tqdm(dl): 
            images, idx = batch 
            desc = self.model(images)
            all_desc[idx] = desc.detach().cpu().numpy().astype(np.float32)

        query_features = all_desc[:len(dataset.query_paths)]
        database_features = all_desc[len(dataset.query_paths):]

        feature_dict = {"query": {"global_desc": query_features}, "database": {"global_desc": database_features}}
        self._save_features(dataset.name, feature_dict)
        self.index = self._setup_index(database_features)
        return feature_dict
    
    def _setup_index(self, desc: np.ndarray) -> faiss.Index:
        desc = np.ascontiguousarray(desc.astype('float32'))  # Ensure float32 and contiguous
        if self.search_dist == 'cosine':
            faiss.normalize_L2(desc)
            index = faiss.IndexFlatIP(desc.shape[1])
        else: 
            index = faiss.IndexFlatL2(desc.shape[1])
        
        index.add(desc)
        return index
    
    def _save_features(self, dataset_name: str, feature_dict: dict) -> None: 
        path = os.path.join(os.path.dirname(__file__), "database", f"{self.name}/{dataset_name}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f: 
            pickle.dump(feature_dict, f)

    def _load_features(self, dataset_name: str): 
        path = os.path.join(os.path.dirname(__file__), "database", f"{self.name}/{dataset_name}.pkl")
        if not os.path.exists(path): 
            return None
        else: 
            with open(path, "rb") as f: 
                feature_dict = pickle.load(f)
                return feature_dict
    
    def match(self, query_features: dict, topk: int=1): 
        query_desc = query_features["global_desc"]
        distances, indices = self.index.search(query_desc, k=topk)
        return indices
    
    def _detect_device(self): 
        if torch.cuda.is_available(): 
            return "cuda"
        else: 
            return "cpu"
        
    def _freeze_model(self, model: nn.Module): 
        for param in model.parameters(): 
            param.requires_grad = False
        return model
        
