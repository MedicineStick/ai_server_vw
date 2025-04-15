from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from models.server_conf import ServerConfig
from models.dsso_model import DSSO_MODEL
from models.dsso_util import CosUploader
import concurrent.futures.thread
import asyncio


class OCR(DSSO_SERVER):
    def __init__(
            self,
            conf:ServerConfig,
            model:DSSO_MODEL,
            executor:concurrent.futures.thread.ThreadPoolExecutor,
            time_blocker:int
            ):
        super().__init__(time_blocker=time_blocker)
        print("--->initialize OCR...")
        self.executor = executor
        self.conf = conf
        self.model = model
        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf

    def dsso_init(self,req:Dict = None)->bool:
        pass

    async def asyn_forward(self, websocket,message):
        import json
        response = await asyncio.get_running_loop().run_in_executor(self.executor, self.dsso_forward, message)
        await websocket.send(json.dumps(response))
    
    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {}
        output = self.model.predict_func_delay(image_url = request["image_url"])
        output_map.update(output)
        output_map['state'] = 'finished'
        return output_map