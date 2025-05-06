

from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig
import os, subprocess
import torch
import urllib
import shutil

class PDF_OCR_Model(DSSO_MODEL):
    def __init__(self, conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        device = torch.device(conf.gpu_id)
        self.if_available = True
        self.conf = conf
        
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
        
        self.conf.olmocr_project_path
        self.conf.olmocr_python_path
        relative_path = os.path.relpath(saved_path,self.conf.olmocr_project_path)

        env = os.environ.copy()
        env["PATH"] = self.conf.olmocr_python_path.strip() +':'+ env["PATH"]
        env["PYTHONPATH"] = self.conf.olmocr_package_path

        cmd = "cd "+self.conf.olmocr_project_path+"; CUDA_VISIBLE_DEVICES="+str(self.conf.olmocr_device_id)+'  '+self.conf.olmocr_python_path.strip()+"/python3  -m olmocr.pipeline ./localworkspace --pdfs "+relative_path
        print(cmd)
        subprocess.run(
                        [cmd],
                        shell=True,
                        env=env
                    )

        return {"result": []}

        
        
        
        
        

        