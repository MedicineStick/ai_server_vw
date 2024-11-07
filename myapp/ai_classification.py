

from typing import Dict
from myapp.dsso_server import DSSO_SERVER
from models.server_conf import ServerConfig
from models.dsso_model import DSSO_MODEL

import concurrent.futures.thread
import asyncio

class AI_Classification(DSSO_SERVER):
    def __init__(self,
                 conf:ServerConfig,
                 model:DSSO_MODEL,
                 executor:concurrent.futures.thread.ThreadPoolExecutor
                 ):
        print("--->initialize AI_Classification...")
        super().__init__()
        self.model = model
        self.conf  = conf
        self.executor = executor

    async def asyn_forward(self, websocket,message):
        import json
        response = await asyncio.get_running_loop().run_in_executor(self.executor, self.dsso_forward, message)
        await websocket.send(json.dumps(response))

    def dsso_init(self,req:Dict = None)->bool:
        pass
        
    def dsso_reload_conf(self,conf:ServerConfig):
        pass

    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {}
        output_map["output"] = self.model.predict_func_delay(image_url = request["image_url"])
        output_map['state'] = 'finished'
        return output_map
