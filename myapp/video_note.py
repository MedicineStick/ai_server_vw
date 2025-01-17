from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from models.server_conf import ServerConfig
from concurrent.futures import ThreadPoolExecutor
import asyncio
from models.dsso_util import CosUploader
import concurrent.futures.thread
from models.dsso_model import DSSO_MODEL
import os
import urllib
import shutil
from models.dsso_util import audio_preprocess, write_json, CosUploader
from moviepy import VideoFileClip
import langid
class VIDEO_NOTE(DSSO_SERVER):
    def __init__(
            self,
            conf:ServerConfig,
            asr_model:DSSO_MODEL,
            llm_model:DSSO_MODEL,
            uploader:CosUploader,
            executor:concurrent.futures.thread.ThreadPoolExecutor,
            time_blocker:int,
            ):
        super().__init__(time_blocker=time_blocker)
        print("--->initialize Motion_Clone_Interface...")
        self.conf = conf
        self.uploader = uploader
        self.executor = executor
        self.time_blocker = time_blocker
        self.asr_model = asr_model
        self.llm_model = llm_model

        self.en_outline_prompt = "generate an outline of the following article"
        self.en_summary_prompt = "generate a summary of the following article"
        self.en_key_words_prompt = "generate some keywords of the following article"

        self.cn_outline_prompt = "针对以下文章生成一段大纲 "
        self.cn_summary_prompt = "针对以下文章生成一段总结 "
        self.cn_key_words_prompt = "针对以下文章生成一些文章关键词 "
        
        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf

    def dsso_init(self,req:Dict = None)->bool:
        pass

    async def asyn_forward(self, websocket,message):
        import json
        response = await asyncio.get_running_loop().run_in_executor(self.executor, self.dsso_forward, message)
        await websocket.send(json.dumps(response))

    def dsso_forward(self, request: Dict) -> Dict:
        print(f"VIDEO_NOTE process started...")
        name = request["name"]
        folder_path = f"./temp/video_note/{name}"
        audio = f"{folder_path}/{name}.wav"
        resampled_audio = f"{folder_path}/{name}1.wav"
        local_video = f"{folder_path}/{name}.mp4"
        cliped_video = f"{folder_path}/{name}_cliped.mp4"
        json_file = f"{folder_path}/{name}.json"
        output_map = {"result":None}
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

        trans_text = "".join([ seg["text"] for seg in segments])
        language, _ = langid.classify(trans_text)

        if language == "zh":
            pass
        elif language == "en":
            pass
        else:
            pass

        

        return output_map