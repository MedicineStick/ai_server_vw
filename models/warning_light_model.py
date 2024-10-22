

from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig

import torch
import sys
sys.path.append("./third_party/ultralytics-main/")
from ultralytics import YOLO
from diffusers.utils import load_image

class WarningLightModel(DSSO_MODEL):
    def __init__(self, conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        device = torch.device(conf.gpu_id)
        self.model = YOLO(conf.warning_light_detection_path).to(device)

    def predict_func(self, **kwargs)->dict:
        image_url = kwargs["image_url"]
        results = self.model.predict(source=load_image(image_url), save=False)
        return results

        