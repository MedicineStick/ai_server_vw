from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from myapp.server_conf import ServerConfig
from myapp.video_generation import Video_Generation
from concurrent.futures import ThreadPoolExecutor


class Video_Generation_Interface(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        super().__init__()
        print("--->initialize Video_Generation_Interface...")
        self.conf = conf
        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf

    def dsso_init(self,req:Dict = None)->bool:
        pass

        
    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {}
        video_model = Video_Generation(self.conf)
        video_model.dsso_init(request)

        # Create a ThreadPoolExecutor with one thread
        flag = True
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit the function to be executed and get a Future object
            future = executor.submit(video_model.dsso_forward, request)
            
            # Wait for the function to complete and get the result
            output_map,flag = future.result()
        
    
        return output_map,flag