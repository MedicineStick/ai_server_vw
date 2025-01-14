
from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from models.server_conf import ServerConfig
from models.dsso_model import DSSO_MODEL
import concurrent.futures.thread
import asyncio
import urllib
import shutil
import os
from models.dsso_util import audio_preprocess, write_json, cut_and_concatenate_video, CosUploader
from moviepy import VideoFileClip
import json

class Fun_Clip(DSSO_SERVER):
    def __init__(self,
                 conf:ServerConfig,
                 asr_model:DSSO_MODEL,
                 uploader:CosUploader,
                 executor:concurrent.futures.thread.ThreadPoolExecutor,
                 time_blocker:int,
                 ):
        super().__init__(time_blocker=time_blocker)
        print("--->initialize warning_light_detection...")
        self.executor = executor
        self.conf = conf
        self.asr_model = asr_model
        self.uploader = uploader

    async def asyn_forward(self, websocket,message):
        import json
        response = await asyncio.get_running_loop().run_in_executor(self.executor, self.dsso_forward, message)
        await websocket.send(json.dumps(response))

    def dsso_init(self,req:Dict = None)->bool:
        pass

    def dsso_reload_conf(self,conf:ServerConfig):
        pass

    def dsso_forward(self, request: Dict) -> Dict:
        print(f"Funclip process started...")
        name = request["name"]
        folder_path = f"./temp/funclip/{name}"
        audio = f"{folder_path}/{name}.wav"
        resampled_audio = f"{folder_path}/{name}1.wav"
        local_video = f"{folder_path}/{name}.mp4"
        cliped_video = f"{folder_path}/{name}_cliped.mp4"
        json_file = f"{folder_path}/{name}.json"
        output_map = {"result":None}

        if int(request["step"]) == 0:
            language = request["language"]
            
            if not os.path.exists(folder_path):
                # Create the folder
                os.makedirs(folder_path)
                print(f"Folder '{folder_path}' created.")
            else:
                print(f"Folder '{folder_path}' already exists.")

            if 'http' in request["input_video"]:
                urllib.request.urlretrieve(request["input_video"], local_video)
            else:
                if os.path.exists(local_video):
                    pass
                else:
                    shutil.copy(request["input_video"], local_video)

            video = VideoFileClip(local_video)
            video.audio.write_audiofile(audio)
            
            print('--->audio_preprocess...')
            audio_preprocess(audio_file=audio,
                        output_audio=resampled_audio,
                        ffmpeg_=self.conf.ai_meeting_ffmpeg_file,
                        sampling_rate_=self.conf.ai_meeting_supported_sampling_rate
                        )


            result = None
            if len(language)==0:
                result = self.asr_model.predict_func_delay(
                                audio = resampled_audio,
                                word_timestamps=True,
                                beam_size = self.conf.funclip_asr_beamsize
                            )
            else:
                result = self.asr_model.predict_func_delay(
                                audio = resampled_audio,
                                word_timestamps=True,
                                language=language,
                                beam_size = self.conf.funclip_asr_beamsize
                            )
            output_map["result"] = result['segments']

            segments = []

            for segment in result['segments']:
                segments.append(
                    {
                        "start":round(segment["start"],2),
                        "end":round(segment["end"],2),
                        "text":segment["text"],
                        }
                    
                    )
            output_map["result"] = segments
            write_json(output_map,json_file)
            return output_map
        
        elif int(request["step"]) == 1:
            if os.path.exists(json_file):
                with open(json_file) as fr:
                    output_map = json.load(fr)
            
            timestamp_list = []

            for index in request["segment_index"]:
                timestamp_list.append( 
                    [
                        output_map["result"][index]["start"],
                        output_map["result"][index]["end"]
                    ]
                     
                     )
            cut_and_concatenate_video(local_video,timestamp_list,cliped_video)
            output_video = self.uploader.upload_video(cliped_video)

            return {"output_url":output_video}
        else:
            return {}


