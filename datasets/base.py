from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import os 
from torchvision import transforms
import numpy as np 
from torch.utils import data

class BaseDataset(ABC):
    @abstractmethod
    def __init__(self, dataset: Dataset):
        pass 

    @abstractmethod 
    def __len__(self) -> int: 
        pass 

    @abstractmethod 
    def __repr__(self) -> str: 
        pass 

    @abstractmethod
    def query(self, idx: int) -> Image:
        pass 

    @abstractmethod
    def query_loader(self, batch_size: int=32, num_workers: int=4) -> DataLoader:
        pass 

    @abstractmethod
    def database_loader(self, batch_size: int=32, num_workers: int=4, transform=None) -> DataLoader:
        pass 


class ImgDataset(data.Dataset): 
    def __init__(self, root: str,paths: list[str], transform: transforms.Compose=None): 
        self.root = root
        self.paths = paths 
        self.transform = transform 

    def __len__(self) -> int: 
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Image: 
        img = Image.open(os.path.join(self.root, self.paths[idx])).convert("RGB")
        if self.transform is not None: 
            img = self.transform(img)
        else: 
            img = transforms.ToTensor()(img)
        return img, idx


class Dataset(BaseDataset): 
    def __init__(self, root: str, name: str, query_paths: str, database_paths: str, ground_truth_path: str): 
        assert os.path.exists(root), f"Root path {root} does not exist"
        assert os.path.exists(query_paths), f"Query image path {query_paths} does not exist"
        assert os.path.exists(database_paths), f"database image path {database_paths} does not exist"
        assert os.path.exists(ground_truth_path), f"Ground truth path {ground_truth_path} does not exist"
        assert name is not None, f"Name is required as argument"

        self.name = name
        self.root = root
        self.query_paths = np.load(query_paths, allow_pickle=True) 
        self.database_paths = np.load(database_paths, allow_pickle=True) 
        self.gt = np.load(ground_truth_path, allow_pickle=True) 
        
        
    def __len__(self) -> int: 
        return len(self.database_paths)

    def __repr__(self) -> str: 
        return f"{self.name}(query_images={len(self.query_paths)}, database_images={len(self.database_paths)})"

    def query(self, idx: int) -> Image:
        return Image.open(os.path.join(self.root, self.query_paths[idx])).convert("RGB")

    def query_loader(self, batch_size: int=32, num_workers: int=4, transform=None) -> DataLoader:
        return data.DataLoader(ImgDataset(self.root, self.query_paths, transform=transform), batch_size=batch_size, num_workers=num_workers, shuffle=False)

    def database_loader(self, batch_size: int=32, num_workers: int=4, transform=None) -> DataLoader:
        return data.DataLoader(ImgDataset(self.root, self.database_paths, transform=transform), batch_size=batch_size, num_workers=num_workers, shuffle=False)

    def ground_truth(self, idx = None): 
        if idx is None: 
            return self.gt
        return self.gt[idx]
