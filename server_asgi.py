
print("--->Loading Mbart Translation...")
from myapp.mbart_translation import mbart_translation
print("--->Loading Realtime_ASR_Whisper_Silero_Vad...")
from myapp.realtime_asr_whisper_silero_vad import Realtime_ASR_Whisper_Silero_Vad
print("--->Loading Realtime_ASR_Whisper_Silero_Vad_Chatbot...")
from myapp.realtime_asr_whisper_silero_vad_chatbot import Realtime_ASR_Whisper_Silero_Vad_Chatbot



import json
import asyncio
import websockets
import concurrent.futures
from models.server_conf import ServerConfig


from models.whisper_large import WhisperLarge
from models.whisper_small import WhisperSmall
from models.mbart_translation_model import MbartTranslationModel
from models.vits_tts_cn import VitsTTSCN
from models.vits_tts_en import VitsTTSEN
from models.silero_vad import Silero_VAD
from models.dsso_util import CosUploader

class WebSocketServer:
    def __init__(self,
                host='localhost',
                port=9502,
                max_workers=40,
                conf_path = "./configs/conf.yaml"
                ):
        self.host = host
        self.port = port
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        global_conf = ServerConfig(conf_path)

        model_dict = {
            "WhisperLarge":WhisperLarge(global_conf),
            "WhisperSmall":WhisperSmall(global_conf),
            "MbartTranslationModel":MbartTranslationModel(global_conf),
            "VitsTTSCN":VitsTTSCN(global_conf),
            "VitsTTSEN":VitsTTSEN(global_conf),
            "SileroVAD":Silero_VAD(global_conf),
            "uploader":CosUploader(global_conf.cos_uploader_mode),

            }
        self.project_name_dict = {


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

                }

    async def handler(self, websocket, path):
        print("--->Connection from {} ".format(websocket.remote_address))
        async for message in websocket:
            task_id_asr = websocket.remote_address[0]+str(websocket.remote_address[1])
            #await self.asr_app.recognize_speech(websocket)
            
            if isinstance(message, str) and 'project_name' in message:
                message_dict = json.loads(message)
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
    port = 9502
    server = WebSocketServer(
        host=host,
        port=port,
        max_workers=max_workers
        )  
    asyncio.run(server.start())

    
    

    
