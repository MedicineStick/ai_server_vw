

from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig
import os
import torch
import urllib
import shutil
from paddleocr import PaddleOCR, draw_ocr

class IMG_OCR_Model(DSSO_MODEL):
    def __init__(self, conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        device = torch.device(conf.gpu_id)
        self.if_available = True
        self.conf = conf
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch") 
    
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


    def predict_func(self, **kwargs)->dict:
        # audio, word_timestamps, language, beam_size,initial_prompt,fp16
        input_image = kwargs["image_url"]
        img_path = "./temp/ocr/"
        img_name = input_image[input_image.rfind('/') + 1:]
        saved_path = os.path.join(img_path,img_name)
        if 'http' in input_image:
                urllib.request.urlretrieve(input_image,saved_path)
        else:
            if os.path.exists(saved_path):
                pass
            else:
                shutil.copy(input_image, saved_path)
        result = self.ocr.ocr(saved_path, cls=True)

        result = result[0]
        if result is None:
            return {"result": []}
        txts1 = [line[1][0]+"        "+ str(round(line[1][1], 3)) for line in result]
        html_path = "temp/ocr/result.html"
        self.strlist_2_html_file(txts1, html_path)

        return {"result": html_path}
        
        

        