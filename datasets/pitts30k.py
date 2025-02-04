from .base import Dataset
import os

class Pitts30k(Dataset):
    def __init__(self,
                root: str,
                name: str="Pitts30k",
                query_img_paths: str=os.path.join(os.path.dirname(__file__), "image_paths", "pitts30k_query.npy"),
                database_img_paths: str=os.path.join(os.path.dirname(__file__), "image_paths", "pitts30k_database.npy"),
                ground_truth_path: str=os.path.join(os.path.dirname(__file__), "image_paths", "pitts30k_gt.npy")):
        super().__init__(name=name, root=root, query_paths=query_img_paths, database_paths=database_img_paths, ground_truth_path=ground_truth_path)

