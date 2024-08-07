import os
from abc import ABC
from pathlib import Path


class Performance(ABC):

    def __init__(self):

        # Cache dir
        self.cache_dir: str = os.path.join(os.path.expanduser("~"), ".cache", 'theatre')
        Path.mkdir(Path(self.cache_dir), parents=True, exist_ok=True)
