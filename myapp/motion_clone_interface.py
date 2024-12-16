from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from models.server_conf import ServerConfig
from myapp.motion_clone import motion_forward
from concurrent.futures import ThreadPoolExecutor
import asyncio
from models.dsso_util import CosUploader
import concurrent.futures.thread


class Motion_Clone_Interface(DSSO_SERVER):
    def __init__(
            self,
            conf:ServerConfig,
            uploader:CosUploader,
            executor:concurrent.futures.thread.ThreadPoolExecutor,
            time_blocker:int
            ):
        super().__init__(time_blocker=time_blocker)
        print("--->initialize Motion_Clone_Interface...")
        self.conf = conf
        self.uploader = uploader
        self.executor = executor
        self.time_blocker = time_blocker
        
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
        # Create a ThreadPoolExecutor with one thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit the function to be executed and get a Future object
            future = executor.submit(motion_forward, self.conf,self.uploader,request)
            
            # Wait for the function to complete and get the result
            output_map = future.result()
        
    
        return output_map