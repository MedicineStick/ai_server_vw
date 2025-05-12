from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from models.server_conf import ServerConfig
from models.dsso_model import DSSO_MODEL
from models.dsso_util import CosUploader,OBS_Uploader
import concurrent.futures.thread
import asyncio
import os,shutil
from pdf2image import convert_from_path
import urllib
class PDF_OCR_V2(DSSO_SERVER):
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
    
    def pdf_2_html(self,pdf_path: str, html_path: str):
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path)
        
        html_content = "<html><head><meta charset='UTF-8'><title>PDF OCR Output</title></head><body>\n"
        
        for i, page in enumerate(pages):
            temp_img_path = f"./temp/ocr/temp_page_{i}.png"
            page.save(temp_img_path, "PNG")

            # Call your external OCR function
            output_map = self.img_model.predict_func_delay(image_url =temp_img_path)

            # Append the OCR result to HTML
            html_content += "<div style='margin-bottom: 20px; white-space: pre-wrap;'>\n"
            for line in output_map["result"]:
                html_content += line + "\n"
            html_content += "</div>\n"

            os.remove(temp_img_path)  # Clean up temporary image

        html_content += "</body></html>"

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)


    def dsso_forward(self, request: Dict) -> Dict:

        input_image = request["image_url"]
        img_path = "./temp/ocr/"
        img_name = input_image[input_image.rfind('/') + 1:]
        saved_path = os.path.join(img_path,img_name)
        html_path = os.path.join(img_path,img_name[:img_name.rfind('.')]+".html")
        if 'http' in input_image:
                urllib.request.urlretrieve(input_image,saved_path)
        else:
            if os.path.exists(saved_path):
                pass
            else:
                shutil.copy(input_image, saved_path)

        output_map = {}
        if saved_path.endswith('.pdf'):
            self.pdf_2_html(pdf_path = saved_path,html_path=html_path)
            output_map["result"] = self.uploader.upload(html_path)
            output_map['state'] = 'finished'
            print("OCR finished",output_map["result"])
        else:
            output_map['state'] = 'error'
            output_map['message'] = 'Unsupported file type. Please upload a pdf file.'
            print("OCR error", output_map["message"])
        return output_map