from abc import ABC, abstractmethod
import threading


class DSSO_MODEL(ABC):
    def __init__(
            self,
            time_blocker:int = 1):
        super().__init__()
        self.if_available = True
        self.time_blocker = time_blocker
        self.lock = threading.Lock()

    @abstractmethod
    def predict_func(self, **kwargs)->dict:
        pass

    def predict_func_delay(self, **kwargs)->dict:

        with self.lock:
            return self.predict_func(**kwargs)




