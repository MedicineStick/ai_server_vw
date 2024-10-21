import sys
sys.path.append("./third_party/ultralytics-main/")
from ultralytics import YOLO
import torch
from PIL import Image
from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
import cv2
from models.server_conf import ServerConfig
from diffusers.utils import load_image
from models.dsso_model import DSSO_MODEL
from PIL import Image

def split_image_pillow(img_src: str, pitch_size: int, img_b: str):
    output_images = []
    # Load the main image
    image = load_image(img_src)
    w, h = image.size
    max_ = max(h, w)
    n_pitch = 0
    image_b = None
    print("w: ",w," h: ",h)
    if max_ % pitch_size == 0:
        n_pitch = max_ // pitch_size
        image_b = image
    else:
        n_pitch = max_ // pitch_size + 1
        new_size = pitch_size * n_pitch
        # Load the background image and resize it
        image_b = Image.open(img_b).resize((new_size, new_size))

        toplx = (new_size - w) // 2
        toply = (new_size - h) // 2

        # Paste the main image onto the resized background
        image_b.paste(image, (toplx, toply))

    # Split the image into smaller pieces
    for i in range(n_pitch):
        for j in range(n_pitch):
            left = j * pitch_size
            upper = i * pitch_size
            right = left + pitch_size
            lower = upper + pitch_size
            img_temp = image_b.crop((left, upper, right, lower))
            output_images.append(img_temp)

    return output_images, n_pitch, w, h

def split_image(img_src:str,pitch_size:int,img_b:str):
    output_images = []
    image = cv2.imread(img_src)
    h, w = image.shape[:2]
    max_ = max(h,w)
    n_pitch = 0
    image_b = None
    if max_%pitch_size==0:
        n_pitch = max_//pitch_size
        image_b = image
    else:
        n_pitch = max_//pitch_size +1
        new_size =pitch_size*n_pitch
        image_b = cv2.imread(img_b)
        image_b = cv2.resize(image_b,(new_size,new_size))

        toplx = (new_size-w)//2
        toply = (new_size-h)//2

        image_b[
                toply:toply+h,
                toplx:toplx+w
                ] = image
    for i in range(0,n_pitch):
        for j in range(0,n_pitch):
            img_temp = image_b[i*pitch_size:(i+1)*pitch_size,j*pitch_size:(j+1)*pitch_size]
            output_images.append(img_temp)
    return output_images,n_pitch


class forgery_detection(DSSO_SERVER):
    def __init__(
        self,
        conf:ServerConfig,
        model:DSSO_MODEL
        ):
        print("--->initialize forgery_detection...")
        super().__init__()
        self.conf = conf
        self._need_mem = conf.forgery_detection_mem
        self.device = torch.device(self.conf.gpu_id)
        self.model = model
        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.forgery_detection_mem

    def dsso_init(self,req:Dict = None)->bool:
        pass
    
    def dsso_forward(self, request: Dict) -> Dict:
        with torch.no_grad():
            
            output_map = {}
            output_map['boxes'] = []
            image_url = request["image_url"]
            image_list,n_pitch, W, H = split_image_pillow(image_url,
                                                          self.conf.forgery_detection_pitch_size,
                                                          self.conf.forgery_detection_b_image
                                                          )
            print(len(image_list),n_pitch)
            for i in range(0,n_pitch):
                for j in range(0,n_pitch):
                    results = self.model.predict_func(image_url = image_list[i*n_pitch+j])
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            if box.conf.cpu().numpy()[0]>self.conf.forgery_detection_threshold:
                                b = list(box.xyxy[0].cpu().numpy())
                                print("b :",i,j,b)

                                conf_out = float(box.conf.cpu().numpy()[0])

                                box_out = [   i*self.conf.forgery_detection_pitch_size+round(b[0]),
                                    j*self.conf.forgery_detection_pitch_size+round(b[1]),
                                    i*self.conf.forgery_detection_pitch_size+round(b[2]),
                                    j*self.conf.forgery_detection_pitch_size+round(b[3])]
                                
                                box_temp = {}
                                box_temp['conf'] = conf_out
                                if box_out[2]<W and box_out[3]<H:
                                    box_temp['box'] = box_out
                                    output_map['boxes'].append(box_temp)
            torch.cuda.empty_cache()
            output_map['state'] = 'finished'
            return output_map,True
