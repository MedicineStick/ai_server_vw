from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from models.server_conf import ServerConfig
from models.dsso_model import DSSO_MODEL
import concurrent.futures.thread
import asyncio
class mbart_translation(DSSO_SERVER):
    def __init__(
            self,
            conf:ServerConfig,
            model:DSSO_MODEL,
            executor:concurrent.futures.thread.ThreadPoolExecutor,
            time_blocker:int
            ):
        super().__init__(time_blocker=time_blocker)
        print("--->initialize mbart_translation...")
        self.executor = executor
        self.conf = conf
        self._need_mem = conf.translation_mem
        self.model = model

    async def asyn_forward(self, websocket,message):
        import json
        response = await asyncio.get_running_loop().run_in_executor(self.executor, self.dsso_forward, message)
        await websocket.send(json.dumps(response))

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
        return output_map
