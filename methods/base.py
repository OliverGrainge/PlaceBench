from abc import ABC, abstractmethod
import torch 
import torch.nn as nn 
from typing import Union, List
from PIL import Image
import numpy as np 
import faiss
from typing import Callable
from torch.utils.data import DataLoader
from tqdm import tqdm 
import pickle
import os


class BaseMethod(ABC):
    @abstractmethod
    def load_database(self, dataset, recompute: bool = False) -> None:
        """Load and prepare the reference database for place recognition.

        Args:
            dataset: Reference dataset to build the database from
            recompute: Whether to force recomputation of the database
        """
        pass
    
    @abstractmethod
    def recognize(self, image: Union[torch.Tensor, Image.Image, DataLoader]) -> List[int]:
        """Recognize places in the input image(s).

        Args:
            image: Input image(s) to recognize

        Returns:
            List of matched reference image indices
        """
        pass

    @abstractmethod
    def extract_features(self, image: Union[torch.Tensor, Image.Image, DataLoader]) -> dict:
        """Extract features from input image(s).

        Args:
            image: Input image(s) to extract features from

        Returns:
            Dictionary containing extracted features
        """
        pass

    @abstractmethod
    def match_features(self, feature: dict, topk: int = 1) -> List[int]:
        """Match extracted features against the reference database.

        Args:
            feature: Dictionary containing features to match
            topk: Number of top matches to return

        Returns:
            List of matched reference image indices
        """
        pass




class SingleStageMethod(BaseMethod):
    def __init__(self, 
                 name: str,
                 model: nn.Module, 
                 feature_dim: int, 
                 transform: Callable, 
                 search_dist: str = 'cosine') -> None:
        """Initialize the single-stage place recognition method.

        Args:
            model: Neural network model for feature extraction
            feature_dim: Dimension of the feature vectors
            transform: Callable to preprocess input images
            search_dist: Distance metric for feature matching ('cosine' or 'euclidean')
        """
        self.name: str = name
        self.model: nn.Module = model
        self.transform: Callable = transform 
        self.feature_dim: int = feature_dim
        self.search_dist: str = search_dist
        self.index: faiss.IndexFlatL2 = None

    def load_database(self, 
                dataset, 
                batch_size: int = 32, 
                num_workers: int = 4, 
                silent: bool = False, 
                device: Union[str, None] = None, 
                recompute: bool = False) -> None:
        
        if self._load_database(dataset.name) and not recompute:
            return
        
        if device is None:
            device = self._detect_device()

        self.model = self.model.to(device)
        self.model.eval()

        dl = dataset.database_loader(batch_size=batch_size, num_workers=num_workers, transform=self.transform)
        database_desc = np.zeros((len(dataset), self.feature_dim), dtype=np.float32)

        try:
            for batch in tqdm(dl, desc="Extracting features from database"):
                batch, idxs = batch
                with torch.no_grad():
                    batch = batch.to(device)
                    global_desc = self.model(batch.to(device))
                database_desc[idxs] = global_desc.cpu().numpy()
        except RuntimeError as e:
            raise RuntimeError(f"Error processing batch: {str(e)}")

        self._setup_database(database_desc, self.search_dist)
        self._save_database(dataset.name)

    def _setup_database(self, database_desc: np.ndarray, search_dist: str) -> None: 
        if search_dist == 'cosine':
            faiss.normalize_L2(database_desc)
            index = faiss.IndexFlatL2(database_desc.shape[1])
            index.add(database_desc)
            self.index = index
        elif search_dist == "euclidean":
            index = faiss.IndexFlatL2(database_desc.shape[1])
            index.add(database_desc)
            self.index = index
        else:
            raise ValueError(f"Unsupported search distance: {search_dist}")
        
    def _save_database(self, dataset_name: str) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f"database/{self.name}/{dataset_name}.faiss")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        faiss.write_index(self.index, file_path)

    def _load_database(self, dataset_name: str) -> bool:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f"database/{self.name}/{dataset_name}.faiss")
        if os.path.exists(file_path):
            self.index = faiss.read_index(file_path)
            return True
        return False

    def recognize(self, input: Union[torch.Tensor, Image.Image, DataLoader], topk: int = 1, recompute: bool = False) -> List[int]:
        features = self.extract_features(input, recompute=recompute)
        matches = self.match_features(features, topk=topk)
        return matches

    def extract_features(self, input: Union[torch.Tensor, Image.Image, 'Dataset'], device: Union[str, None] = None, recompute: bool = False) -> dict:
        if device is None:
            device = self._detect_device()

        self.model = self.model.to(device)
        self.model.eval()
        
        if isinstance(input, Image.Image):
            input = self.transform(input).unsqueeze(0).to(device)
            with torch.no_grad():
                features = self.model(input)
            return {"global_desc": features.cpu().numpy()}
        
        elif isinstance(input, torch.Tensor):
            input = input.to(device)
            with torch.no_grad():
                features = self.model(input)
            return {"global_desc": features.cpu().numpy()}
    
        elif hasattr(input, '__class__') and 'datasets' in str(type(input)):
            if not recompute and self._load_features(input.name):
                return self.query_desc
            dl = input.query_loader(batch_size=32, num_workers=4, transform=self.transform)
            desc = np.zeros((len(dl.dataset.paths), self.feature_dim), dtype=np.float32)
            with torch.no_grad():
                for batch in tqdm(dl, desc="Extracting features from query"):
                    batch, idxs = batch
                    global_desc = self.model(batch.to(device))
                    desc[idxs] = global_desc.cpu().numpy()
            
            desc = {"global_desc": desc}
            self._save_features(desc, input.name)
            return desc
        raise ValueError(f"Unsupported input type: {type(input)}")
    
    def _save_features(self, query_desc: dict, dataset_name: str) -> None: 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f"database/{self.name}/{dataset_name}_features.pkl")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(query_desc, f)

    def _load_features(self, dataset_name: str) -> bool:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f"database/{self.name}/{dataset_name}_features.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                self.query_desc = pickle.load(f)
            return True
        return False

    def match_features(self, query_desc: dict, topk: int=1) -> List[int]: 
        assert self.index is not None, "Database not loaded. Call load_database() first."
        query_desc = query_desc["global_desc"]
        if self.search_dist == 'cosine': 
            faiss.normalize_L2(query_desc)
            D, I = self.index.search(query_desc, topk)
        elif self.search_dist == 'euclidean': 
            D, I = self.index.search(query_desc, topk)
        return I

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"



