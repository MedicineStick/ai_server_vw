

from typing import Dict
import pynvml
from abc import ABC, abstractmethod
from models.server_conf import ServerConfig
import time
def get_gpu_mem_info(gpu_id:int)->(float,float,float):
    pynvml.nvmlInit()
    if gpu_id<0 or gpu_id>=pynvml.nvmlDeviceGetCount():
        print("gpu {} doesn't exist!\n".format(gpu_id))
        return 0.,0.,0.
    
    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total/1024/1024,2)
    used = round(meminfo.used/1024/1024,2)
    free = round(meminfo.free/1024/1024,2)
    return total,used,free

def check_if_forward(mem:float,gpu_id:int)->bool:

    total,allocate,free = get_gpu_mem_info(gpu_id)
    print("total : {}, allocate: {}, free: {}, need: {}".format(total,allocate,free,mem))
    if mem>= free:
        return False
    else:
        return True 

class DSSO_SERVER(ABC):
    def __init__(
            self,
            time_blocker:int,
            ):
        super().__init__()
        self._need_mem = 0.0
        self._available = True
        self._time_blocker = time_blocker

    def load_args(self, dsso_args:Dict[str,str]):
        for k,v in dsso_args.items():
            k = k.strip()
            v = v.strip()

            if k !="":
                self._dsso_args[k] = v
            else:
                pass

    def print_args(self):
        for k,v in self._dsso_args.items():
            print("{} : {} \n".format(k,v))

    def set_max_gpu_mem(self,max_gpu_mem:float):
        if max_gpu_mem>0:
            self._max_gpu_mem = max_gpu_mem
        else:
            self._max_gpu_mem = 0.
    
    def set_need_mem(self,need_mem:float):
        if need_mem>0:
            self._need_mem = need_mem
        else:
            self._need_mem = 0.
            
    @abstractmethod
    async def asyn_forward(self, websocket,message):
        pass

    async def asyn_forward_with_locker(self, websocket,message):
        while True:
            if self._available:
                self._available = False
                try:
                    await self.asyn_forward(websocket,message)
                except Exception as error:
                    print('Error: ', error)
                finally:
                    self._available = True
                    break
            else:
                time.sleep(self._time_blocker)

    @abstractmethod
    def dsso_forward(self,req:Dict)->Dict:
        pass
    
    def dsso_forward_with_locker(self,req:Dict)->Dict:
        self._available = False
        output = {}
        output['state'] = ""
        """
        output,flag = self.dsso_forward(req)
        output['state'] = 'finished'
        self._available = True
        return output,flag
        """
        try:
            output = self.dsso_forward(req)
        except Exception as error:
            print('Error: ', error)
            output['state'] += str(error)
        finally:
            self._available = True
            return output

    @abstractmethod
    def dsso_init(self,req:Dict = None)->bool:
        pass

    def if_available(self)->bool:
        return self._available

    @abstractmethod
    def dsso_reload_conf(self,conf:ServerConfig):
        pass
