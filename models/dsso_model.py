from abc import ABC, abstractmethod
from typing import Any
from models.server_conf import ServerConfig


class DSSO_MODEL(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict_func(self, **kwargs)->dict:
        pass



