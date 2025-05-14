from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from models.server_conf import ServerConfig
from models.dsso_model import DSSO_MODEL
from models.dsso_util import CosUploader,OBS_Uploader
import concurrent.futures.thread
import asyncio


class IMG_OCR(DSSO_SERVER):
    def __init__(
            self,
            conf:ServerConfig,
            img_model:DSSO_MODEL,
            uploader:OBS_Uploader,
            executor:concurrent.futures.thread.ThreadPoolExecutor,
            time_blocker:int
            ):
        super().__init__(time_blocker=time_blocker)
        print("--->initialize OCR...")
        self.executor = executor
        self.conf = conf
        self.img_model = img_model
        self.uploader = uploader
        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf

    def dsso_init(self,req:Dict = None)->bool:
        pass

    async def asyn_forward(self, websocket,message):
        import json
        response = await asyncio.get_running_loop().run_in_executor(self.executor, self.dsso_forward, message)
        await websocket.send(json.dumps(response))
    
    def strlist_2_html_file(
            self,
            list1: list[str],
            html_path: str
            ):
        html_content = "<html>\n<head><meta charset='UTF-8'><title>String List</title></head>\n<body>\n<ul>\n"
        for item in list1:
            html_content += f"  <li>{item}</li>\n"
        html_content += "</ul>\n</body>\n</html>"

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {}
        if request["image_url"].endswith('.jpg') or request["image_url"].endswith('.png'):
            output = self.img_model.predict_func_delay(image_url = request["image_url"])
            html_path = "temp/ocr/result.html"
            self.strlist_2_html_file(output["result"], html_path)
            output_map["html_path"] = self.uploader.upload(html_path)
            output_map["result"] = '\n'.join(output["result"])
            output_map['state'] = 'finished'
            print("OCR finished",output_map["html_path"])
        else:
            output_map['state'] = 'error'
            output_map['message'] = 'Unsupported file type. Please upload a .jpg or .png image.'
            print("OCR error", output_map["message"])
        return output_map