import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from myapp.dsso_server import DSSO_SERVER
from myapp.server_conf import ServerConfig
from typing import Dict
from diffusers.utils import load_image
from myapp.dsso_util import load_image_cv2,CosUploader

class Sam1(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        print("--->initialize Sam1...")
        super().__init__()
        self.conf = conf
        self.device = torch.device(self.conf.gpu_id)
        self.sam = sam_model_registry[self.conf.sam1_model_type](checkpoint=self.conf.sam1_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

        try:
            self.uploader = CosUploader(self.conf.super_resolution_mode)
        except Exception as e:
            self.uploader = None

    def dsso_init(self,req:Dict = None)->bool:

        pass
        
    
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = self.conf.ai_classification_mem
        self.device = torch.device(self.conf.gpu_id)


    def dsso_forward(self, request: Dict) -> Dict:
        with torch.no_grad():
            image_url  = request["image_url"]
            self.predictor.set_image(load_image_cv2(image_url))
            image_embedding = self.predictor.get_image_embedding().cpu().numpy()
            np.save("./temp/output.npy",image_embedding)
            url = self.uploader.upload_file(file_temp="./temp/output.npy",key=None)
            output_map = {"npy_url":url}
            output_map['state'] = 'finished'
            return output_map,True
