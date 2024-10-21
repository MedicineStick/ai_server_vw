print("--->Loading AI_Classification ...")
from myapp.ai_classification import AI_Classification
print("--->Loading forgery_detection ...")
from myapp.forgery_detection import forgery_detection
print("--->Loading warning_light_detection ...")
from myapp.warning_light_detection import warning_light_detection
print("--->Loading Super_Resolution ...")
from myapp.super_resolution import Super_Resolution
print("--->Loading AI_Meeting_Chatbot ...")
from myapp.ai_meeting_chatbot import AI_Meeting_Chatbot
print("--->Loading Mbart Translation...")
from myapp.mbart_translation import mbart_translation
print("--->Loading Video_Generation...")
from myapp.video_generation_interface import Video_Generation_Interface
print("--->Loading Super_Resolution_Video...")
from myapp.super_resulution_video import Super_Resolution_Video
print("--->Loading Realtime_ASR_Whisper_Silero_Vad...")
from myapp.realtime_asr_whisper_silero_vad import Realtime_ASR_Whisper_Silero_Vad
print("--->Loading Sam2...")
from myapp.sam2 import Sam2
print("--->Loading Sam1...")
from myapp.sam1 import Sam1


import json
import torch
import asyncio
import websockets
import time
import concurrent.futures
import logging
from models.server_conf import ServerConfig

from models.ai_classification_model import AiClassificationModel
from models.warning_light_model import WarningLightModel
from models.whisper_large import WhisperLarge
from models.whisper_small import WhisperSmall
from models.forgery_detection_model import ForgeryDetectionModel
from models.mbart_translation_model import MbartTranslationModel
from models.vits_tts_cn import VitsTTSCN
from models.vits_tts_en import VitsTTSEN
from models.esr_gan import ESRGan
from models.dsso_util import CosUploader

conf_path = "./configs/conf.yaml"

global_conf = ServerConfig(conf_path)

model_dict = {
    "AiClassificationModel" : AiClassificationModel(global_conf),
    "WarningLightModel" : WarningLightModel(global_conf),
    "WhisperLarge":WhisperLarge(global_conf),
    "WhisperSmall":WhisperSmall(global_conf),
    "ForgeryDetectionModel":ForgeryDetectionModel(global_conf),
    "MbartTranslationModel":MbartTranslationModel(global_conf),
    "VitsTTSCN":VitsTTSCN(global_conf),
    "VitsTTSEN":VitsTTSEN(global_conf),
    "ESRGan":ESRGan(global_conf),
    "uploader":CosUploader(global_conf.cos_uploader_mode)
    }

Model_name_dict = {
                "warning_light_detection":warning_light_detection(
                    global_conf,
                    model_dict["WarningLightModel"]
                    ),

                "ai_meeting_assistant_chatbot":AI_Meeting_Chatbot(
                    global_conf,
                    asr_model=model_dict["WhisperLarge"],
                    translation_model = model_dict["MbartTranslationModel"],
                    uploader=model_dict["uploader"]
                    ),

                "ai_classification":AI_Classification(
                    global_conf,
                    model_dict["AiClassificationModel"]
                    ),

                "forgery_detection":forgery_detection(
                    global_conf,
                    model_dict["ForgeryDetectionModel"]
                    ),

                "super_resolution":Super_Resolution(
                    global_conf,
                    model=model_dict["ESRGan"],
                    uploader=model_dict["uploader"]
                    ),
                "translation":mbart_translation(
                    global_conf,
                    model=model_dict["MbartTranslationModel"],
                    ),
                    
                "Video_Generation_Interface":Video_Generation_Interface(global_conf),
                "super_resulution_video":Super_Resolution_Video(global_conf),
                "realtime_asr_whisper":Realtime_ASR_Whisper_Silero_Vad(
                    global_conf,
                    model_dict["WhisperSmall"],
                    translation_model = model_dict["MbartTranslationModel"],
                    ),
                "sam2":Sam2(global_conf),
                "sam1":Sam1(global_conf)
            }

time_blocker = 10    


def http_inference(global_conf, message,model_name):
    global Model_name_dict
    flag = False
    while True:
        if Model_name_dict[model_name].if_available():
            with torch.no_grad(): 
                output = {}
                Model_name_dict[model_name].dsso_reload_conf(global_conf)
                Model_name_dict[model_name].dsso_init(message)
                output,flag= Model_name_dict[model_name].dsso_forward_http(message)
                print("--->Finish processing {}...".format(model_name))
                break
        else:
            print("--->Model {} is a little bit of busy right now, please wait...".format(Model_name_dict[model_name]))
            time.sleep(time_blocker)

    return output,flag

def realtime_asr_inference(message,online_asr_model):
    output = {}
    with torch.no_grad():
        output,stop = online_asr_model.dsso_forward(message)
    return output,stop

def realtime_asr_whisper_inference(message,model_name):
    global Model_name_dict
    output = {}
    with torch.no_grad():
        #Model_name_dict[model_name].dsso_init(message)
        output,flag= Model_name_dict[model_name].dsso_forward_http(message)
        return output,flag

async def start_server(websocket, path):
    with torch.no_grad(): 
        global pool,global_conf,Model_name_dict
        loop = asyncio.get_running_loop()
        global_conf = ServerConfig(conf_path)
        print("--->Connection from {} ".format(websocket.remote_address))
        task_id_asr = websocket.remote_address[0]+str(websocket.remote_address[1])
        while True:
            message = await websocket.recv()
            if isinstance(message, str) and 'project_name' in message:
                message_dict = json.loads(message)
                model_name = message_dict['project_name']
                #message_dict["task_id"] = task_id
                
                if model_name not in Model_name_dict.keys():
                    print("--->Can't recognize model name : {} \n".format(model_name))
                    break
                else:
                    if model_name=="realtime_asr_whisper":
                        message_dict["task_id"] = task_id_asr
                        response, stop = await loop.run_in_executor(pool, realtime_asr_whisper_inference, message_dict,model_name)
                        if response["if_send"]:
                            await websocket.send(json.dumps(response))
                        else:
                            pass
                        if stop: break
                    else:
                        print("request: ",message_dict)
                        response, _ = await loop.run_in_executor(pool, http_inference, global_conf, message_dict,model_name)
                        print("response: ",response)
                        await websocket.send(json.dumps(response))
                        break

async def start(max_workers:int,interface,port):
    print('--->Start server! -_-')
    global pool

    logging.basicConfig(level=logging.INFO)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers)

    async with websockets.serve(start_server, interface, port):
        await asyncio.Future()


if __name__ == '__main__':
    
    max_workers = 40
    interface =  '0.0.0.0'
    #port = int(sys.argv[1]) #9501
    port = 9501
    asyncio.run(start(max_workers,interface,port))

    
