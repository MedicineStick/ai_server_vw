import sys
sys.path.append("../3.forgery_detection/ultralytics-main")

import torch
from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from myapp.server_conf import ServerConfig
import base64
sys.path.append("../../softwares/vits-main")
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence




def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

class voice_conversion(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        super().__init__()
        self.conf = conf
        self._need_mem = conf.forgery_detection_mem
        self.device = torch.device(self.conf.gpu_id)

        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.forgery_detection_mem

    def dsso_init(self,req:Dict = None)->bool:
        pass

        
    def dsso_forward(self, request: Dict) -> Dict:

        output_map = {}
        output_map['state'] = 'finished'
        return output_map,True
