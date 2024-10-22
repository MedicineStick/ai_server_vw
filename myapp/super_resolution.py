from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from models.server_conf import ServerConfig
from models.dsso_model import DSSO_MODEL
from models.dsso_util import CosUploader

class Super_Resolution(DSSO_SERVER):
    def __init__(
            self,
            conf:ServerConfig,
            model:DSSO_MODEL,
            uploader:CosUploader
            ):
        super().__init__()
        print("--->initialize Super_Resolution...")
        self.conf = conf
        self._need_mem = self.conf.super_resolution_mem
        self.model = model
        self.uploader = uploader
        
    def dsso_init(self,message:Dict) -> bool:        
        pass

    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.super_resolution_mem

    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {}
        output = self.model.predict_func_delay(image_url = request["image_url"])
        url1 = self.uploader.upload_image(output["image1"])
        url2 = self.uploader.upload_image(output["image2"])
        output_map["output_image_url"] = url1
        output_map["download_url"] = url2
        output_map['state'] = 'finished'
        return output_map,True
 