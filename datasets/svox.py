import os

from .base import VPRDataset


class SVOX(VPRDataset):
    def __init__(
        self,
        root: str,
        name: str = "SVOX",
        query_paths: str = None,
        database_paths: str = None,
        ground_truth_path: str = None,
        condition: str = "sun",
    ):
        
        name = f"{name}-{condition.lower()}"
        ground_truth_path = os.path.join(
            os.path.dirname(__file__), "image_paths", f"svox_{condition}_gt.npy"
        )
        query_paths = os.path.join(
            os.path.dirname(__file__), "image_paths", f"svox_{condition}_query.npy"
        )
        database_paths = os.path.join(
            os.path.dirname(__file__), "image_paths", f"svox_database.npy"
        )
        super().__init__(
            name=name,
            root=root,
            query_paths=query_paths,
            database_paths=database_paths,
            ground_truth_path=ground_truth_path,
        )
