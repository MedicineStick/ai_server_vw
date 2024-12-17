
from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from models.server_conf import ServerConfig
from models.dsso_model import DSSO_MODEL
import concurrent.futures.thread
import asyncio
import urllib
import shutil
import os
from models.dsso_util import audio_preprocess
from moviepy.editor import VideoFileClip
class Fun_Clip(DSSO_SERVER):
    def __init__(self,
                 conf:ServerConfig,
                 asr_model:DSSO_MODEL,
                 llm_model:DSSO_MODEL,
                 executor:concurrent.futures.thread.ThreadPoolExecutor,
                 time_blocker:int,
                 ):
        super().__init__(time_blocker=time_blocker)
        print("--->initialize warning_light_detection...")
        self.executor = executor
        self.conf = conf
        self.asr_model = asr_model
        self.llm_model = llm_model

    async def asyn_forward(self, websocket,message):
        import json
        response = await asyncio.get_running_loop().run_in_executor(self.executor, self.dsso_forward, message)
        await websocket.send(json.dumps(response))

    def dsso_init(self,req:Dict = None)->bool:
        pass

    def dsso_reload_conf(self,conf:ServerConfig):
        pass

    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {"video":""}
        language = request["language"]
        name = request["name"]
        audio = f"./temp/funclip/{name}.wav"
        resampled_audio = f"./temp/funclip/{name}1.wav"
        local_video = f"./temp/funclip/{name}.mp4"
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
                    "start":segment["start"],
                    "end":segment["end"],
                    "text":segment["text"],
                    }
                
                )
        output_map["result"] = segments
        return output_map
