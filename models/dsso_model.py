from abc import ABC, abstractmethod
import time


class DSSO_MODEL(ABC):
    def __init__(
            self,
            time_blocker:int = 1):
        super().__init__()
        self.if_available = True
        self.time_blocker = time_blocker

    @abstractmethod
    def predict_func(self, **kwargs)->dict:
        pass

    def predict_func_delay(self, **kwargs)->dict:
        while not self.if_available:
            time.sleep(self.time_blocker)
        results = None
        try:
            self.if_available = False
            results = self.predict_func(**kwargs)
            self.if_available = True
        except Exception as e:
            print(e)
            self.if_available = True
        return results



