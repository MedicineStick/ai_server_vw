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
#print("--->Loading Video_Generation...")
#from myapp.video_generation_interface import Video_Generation_Interface
print("--->Loading Super_Resolution_Video...")
#from myapp.super_resulution_video import Super_Resolution_Video
print("--->Loading Realtime_ASR_Whisper_Silero_Vad...")
from myapp.realtime_asr_whisper_silero_vad import Realtime_ASR_Whisper_Silero_Vad
#print("--->Loading Sam2...")
#from myapp.sam2 import Sam2
print("--->Loading Sam1...")
from myapp.sam1 import Sam1
print("--->Loading Realtime_ASR_Whisper_Silero_Vad_Chatbot...")
from myapp.realtime_asr_whisper_silero_vad_chatbot import Realtime_ASR_Whisper_Silero_Vad_Chatbot
#print("--->Loading Motion_Clone...")
#from myapp.motion_clone import Motion_Clone
#print("--->Loading Jumper_Cutter...")
#from myapp.jumper_cutter import Jumper_Cutter
#print("--->Loading Fun_Clip...")
#from myapp.funclip import Fun_Clip
#print("--->Loading VIDEO_NOTE...")
#from myapp.video_note import VIDEO_NOTE


import json
import asyncio
import websockets
import concurrent.futures
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
from models.gfp_gan import GFPGan
from models.silero_vad import Silero_VAD
from models.dsso_llm import DssoLLM
from models.dsso_util import CosUploader

class WebSocketServer:
    def __init__(self,
                host='localhost',
                port=9501,
                max_workers=40,
                conf_path = "./configs/conf.yaml"
                ):
        self.host = host
        self.port = port
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
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
            "GFPGan":GFPGan(global_conf),
            "SileroVAD":Silero_VAD(global_conf),
            "DssoLLM":DssoLLM(global_conf),
            "uploader":CosUploader(global_conf.cos_uploader_mode),

            }
        self.project_name_dict = {
                "warning_light_detection":warning_light_detection(
                    global_conf,
                    model_dict["WarningLightModel"],
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                    ),

                "ai_meeting_assistant_chatbot":AI_Meeting_Chatbot(
                    global_conf,
                    asr_model=model_dict["WhisperLarge"],
                    translation_model = model_dict["MbartTranslationModel"],
                    vad_model=model_dict["SileroVAD"],
                    llm_model=model_dict["DssoLLM"],
                    uploader=model_dict["uploader"],
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                    ),

                "ai_classification":AI_Classification(
                    global_conf,
                    model_dict["AiClassificationModel"],
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                    ),

                "forgery_detection":forgery_detection(
                    global_conf,
                    model_dict["ForgeryDetectionModel"],
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                    ),

                "super_resolution":Super_Resolution(
                    global_conf,
                    model=model_dict["GFPGan"],
                    uploader=model_dict["uploader"],
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                    ),
                "translation":mbart_translation(
                    global_conf,
                    model=model_dict["MbartTranslationModel"],
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                    ),
                "realtime_asr_whisper":Realtime_ASR_Whisper_Silero_Vad(
                    global_conf,
                    model_dict["WhisperSmall"],
                    vad_model=model_dict["SileroVAD"],
                    translation_model = model_dict["MbartTranslationModel"],
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                    ),
                "realtime_asr_whisper_chatbot":Realtime_ASR_Whisper_Silero_Vad_Chatbot(
                    global_conf,
                    asr_model = model_dict["WhisperSmall"],
                    vad_model=model_dict["SileroVAD"],
                    llm_model=model_dict["DssoLLM"],
                    cn_tts_model=model_dict["VitsTTSCN"],
                    en_tts_model=model_dict["VitsTTSEN"],
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                    ),
                "sam1":Sam1(global_conf,
                            executor=self.executor,
                            time_blocker=global_conf.time_blocker),
                
                }

    """
                "super_resulution_video":Super_Resolution_Video(
                    global_conf,
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                    ),
                "sam2":Sam2(global_conf,
                            executor=self.executor,
                            time_blocker=global_conf.time_blocker),
                "motion_clone":Motion_Clone(
                    conf=global_conf,
                    uploader=model_dict["uploader"],
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                ),
                "jumper_cutter":Jumper_Cutter(
                    conf=global_conf,
                    uploader=model_dict["uploader"],
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                ),
                "fun_clip":Fun_Clip(
                    global_conf,
                    asr_model = model_dict["WhisperLarge"],
                    uploader=model_dict["uploader"],
                    executor=self.executor,
                    time_blocker=global_conf.time_blocker,
                    ),
                
                "video_note":VIDEO_NOTE(
                        global_conf,
                        asr_model=model_dict["WhisperLarge"],
                        llm_model=model_dict["DssoLLM"],
                        uploader=model_dict["uploader"],
                        executor=self.executor,
                        time_blocker=global_conf.time_blocker,
                    ),
                """
                

    async def handler(self, websocket, path):
        print("--->Connection from {} ".format(websocket.remote_address))
        async for message in websocket:
            task_id_asr = websocket.remote_address[0]+str(websocket.remote_address[1])
            #await self.asr_app.recognize_speech(websocket)
            
            if isinstance(message, str) and 'project_name' in message:
                message_dict = json.loads(message)
                if "task_id" in message_dict.keys():
                    pass
                else:
                    message_dict['task_id'] = task_id_asr
                model_name = message_dict['project_name']
                if "asr" in model_name:
                    pass
                else:
                    print(message_dict)
                await self.project_name_dict[model_name].asyn_forward_with_locker(websocket,message_dict)
            
    async def start(self):
        print('--->Start server! -_-')
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # Run forever
            

if __name__ == '__main__':
    

    max_workers = 40
    host =  '0.0.0.0'
    port = 9501
    server = WebSocketServer(
        host=host,
        port=port,
        max_workers=max_workers
        )  
    asyncio.run(server.start())

    
    

    
