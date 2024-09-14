import sys
sys.path.append("../3.forgery_detection/ultralytics-main")

from ultralytics import YOLO
import torch
from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from myapp.server_conf import ServerConfig
from diffusers.utils import load_image


def get_VOC_Decription_MAP_1()->(dict,dict):

    VOC_CLASSES_MAP = {}
    file_ = open("./myapp/resource/0_warning_light/label_map.txt",mode='r')
    lines_ = file_.readlines()
    file_.close()

    for line_ in lines_:
        line_ = line_.strip().split("\t")
        VOC_CLASSES_MAP[int(line_[1].strip())] = line_[0].strip()+'.png'

    VOC_Decription_MAP = {}
    file_ = open("./myapp/resource/0_warning_light/Warning_Light_Decription.txt",mode='r')
    lines_ = file_.readlines()
    file_.close()
    for line_ in lines_:
        line_ = line_.strip().split("\t")
        name = line_[0].strip()
        desciption = line_[1].strip()

        if name in VOC_Decription_MAP.keys():
            VOC_Decription_MAP[name].add(desciption)
        else:
            VOC_Decription_MAP[name] = set()
            VOC_Decription_MAP[name].add(desciption)

    VOC_Decription_MAP_1 = {}
    for k,v in VOC_Decription_MAP.items():
        listv = list(v)
        VOC_Decription_MAP_1[k] = '#'.join(listv)


    for k,v in VOC_CLASSES_MAP.items():
        if v in VOC_Decription_MAP_1.keys():
            pass
        else:
            print("Doesn't exist: ",k,v)
    return VOC_CLASSES_MAP,VOC_Decription_MAP_1


class warning_light_detection(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        super().__init__()
        print("--->initialize warning_light_detection...")
        self.conf = conf
        self.device = torch.device(self.conf.gpu_id)
        self._need_mem = self.conf.warning_light_detection_mem
        self.model = YOLO(self.conf.warning_light_detection_path).to(self.device)  # load an official model
        

    def dsso_init(self,req:Dict = None)->bool:
        pass

    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.forgery_detection_mem
        self.VOC_CLASSES_LIST = []
        for i in range(0,conf.warning_light_detection_class_num):
            self.VOC_CLASSES_LIST.append(str(i)) 

    def dsso_forward(self, request: Dict) -> Dict:
        with torch.no_grad():
            output_map = {}
            image_url = request["image_url"]
            output_map['output'] = {}
            results = self.model.predict(source=load_image(image_url), save=False)
            VOC_CLASSES_MAP,VOC_Decription_MAP_1 = get_VOC_Decription_MAP_1()
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if box.conf.cpu().numpy()[0]>self.conf.warning_light_detection_threshold:
                        #b = list(box.xyxy[0].cpu().numpy())
                        light = int(box.cls.cpu().numpy()[0])
                        #conf_out = float(box.conf.cpu().numpy()[0])
                        output_map['output'][VOC_CLASSES_MAP[light]] = VOC_Decription_MAP_1[VOC_CLASSES_MAP[light]]
            torch.cuda.empty_cache()
            output_map['state'] = 'finished'
            return output_map,True
