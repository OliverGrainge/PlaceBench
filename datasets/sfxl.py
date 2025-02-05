import os

from .base import VPRDataset


class SFXL(VPRDataset):
    def __init__(
        self,
        root: str,
        name: str = "SFXL",
        query_paths: str = os.path.join(
            os.path.dirname(__file__), "image_paths", "sfxl_query.npy"
        ),
        database_paths: str = os.path.join(
            os.path.dirname(__file__), "image_paths", "sfxl_database.npy"
        ),
        ground_truth_path: str = os.path.join(
            os.path.dirname(__file__), "image_paths", "sfxl_gt.npy"
        ),
    ):
        super().__init__(
            name=name,
            root=root,
            query_paths=query_paths,
            database_paths=database_paths,
            ground_truth_path=ground_truth_path,
        )
