import torch
from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from models.server_conf import ServerConfig
from models.dsso_model import DSSO_MODEL
class mbart_translation(DSSO_SERVER):
    def __init__(
            self,
            conf:ServerConfig,
            model:DSSO_MODEL
            ):
        super().__init__()
        print("--->initialize mbart_translation...")
        self.conf = conf
        self._need_mem = conf.translation_mem
        self.model = model

    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.translation_mem

    def dsso_init(self,req:Dict = None)->bool:
        pass

        
    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {}
        task = request["task"]
        text = request["text"]
        
        output_map['result'] = self.model(task= task,text = text)
        output_map['state'] = 'finished'
        return output_map,True
