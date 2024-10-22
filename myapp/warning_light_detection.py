import sys
sys.path.append("./third_party/ultralytics-main/")
from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from models.server_conf import ServerConfig
from models.dsso_model import DSSO_MODEL

def get_VOC_Decription_MAP_1()->tuple[dict,dict]:

    VOC_CLASSES_MAP = {}
    file_ = open("./checkpoints/warning_light/label_map.txt",mode='r')
    lines_ = file_.readlines()
    file_.close()

    for line_ in lines_:
        line_ = line_.strip().split("\t")
        VOC_CLASSES_MAP[int(line_[1].strip())] = line_[0].strip()+'.png'

    VOC_Decription_MAP = {}
    file_ = open("./checkpoints/warning_light/Warning_Light_Decription.txt",mode='r')
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
    def __init__(self,
                 conf:ServerConfig,
                 model:DSSO_MODEL
                 ):
        super().__init__()
        print("--->initialize warning_light_detection...")
        self.conf = conf
        self.model = model

    def dsso_init(self,req:Dict = None)->bool:
        pass

    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.forgery_detection_mem
        self.VOC_CLASSES_LIST = []
        for i in range(0,conf.warning_light_detection_class_num):
            self.VOC_CLASSES_LIST.append(str(i)) 

    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {}
        output_map['output'] = {}
        results = self.model.predict_func_delay(image_url = request["image_url"])
        VOC_CLASSES_MAP,VOC_Decription_MAP_1 = get_VOC_Decription_MAP_1()
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf.cpu().numpy()[0]>self.conf.warning_light_detection_threshold:
                    #b = list(box.xyxy[0].cpu().numpy())
                    light = int(box.cls.cpu().numpy()[0])
                    #conf_out = float(box.conf.cpu().numpy()[0])
                    output_map['output'][VOC_CLASSES_MAP[light]] = VOC_Decription_MAP_1[VOC_CLASSES_MAP[light]]
        output_map['state'] = 'finished'
        return output_map,True
