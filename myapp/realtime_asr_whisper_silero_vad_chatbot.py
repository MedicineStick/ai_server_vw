from typing import Dict
import numpy as np
import torch
import base64
from myapp.dsso_server import DSSO_SERVER
from models.server_conf import ServerConfig
from models.dsso_util import process_timestamps
import torchaudio
# https://github.com/snakers4/silero-vad/blob/master/examples/cpp/silero-vad-onnx.cpp
# https://github.com/mozilla/DeepSpeech-examples/blob/r0.8/mic_vad_streaming/mic_vad_streaming.py
# https://github.com/snakers4/silero-vad/blob/master/examples/microphone_and_webRTC_integration/microphone_and_webRTC_integration.py
# https://github.com/davabase/whisper_real_time/tree/master
import sys
sys.path.append("/home/tione/notebook/lskong2/projects/2.tingjian/")
from models.dsso_model import DSSO_MODEL
import torchaudio
import re

class Realtime_ASR_Whisper_Silero_Vad_Chatbot(DSSO_SERVER):
    def __init__(
            self,
            conf:ServerConfig,
            asr_model:DSSO_MODEL,
            vad_model:DSSO_MODEL,
            ):
        print("--->initialize Realtime_ASR_Whisper_Silero_Vad_Chatbot...")
        super().__init__()
        self.conf = conf
        self._need_mem = self.conf.online_asr_mem
        self.model = None
        self.audio_tensors:dict[str:torch.Tensor] = {}
        self.output_table:dict[str:dict] = {}
        self.realtime_asr_model_sample = self.conf.realtime_asr_model_sample
        self.realtime_asr_gap_ms = self.conf.realtime_asr_gap_ms
        self.realtime_asr_beam_size = self.conf.realtime_asr_beam_size
        self.realtime_asr_min_silence_duration_ms_chatbot = self.conf.realtime_asr_min_silence_duration_ms_chatbot
        self.realtime_asr_max_length_ms_chatbot = self.conf.realtime_asr_max_length_ms_chatbot
        self.realtime_asr_adaptive_thresholding_chatbot = self.conf.realtime_asr_adaptive_thresholding_chatbot
        self.vad_model = vad_model
        self.asr_model = asr_model
    
    def asr_forward(
            self,
            valid_tensor:torch.Tensor,
            request:dict,
            )->str:
        result = ""
        if request["language_code"]=="zh":
            initial_prompt = "以下是普通话的句子，这是一段会议记录。"

            result = self.asr_model.predict_func_delay(
                audio = valid_tensor,
                language=request['language_code'],
                initial_prompt=initial_prompt,
                beam_size = self.realtime_asr_beam_size,
                fp16=torch.cuda.is_available(),
            )

        else:
            result = self.asr_model.predict_func_delay(
                    audio = valid_tensor, 
                    language=request['language_code'],
                    beam_size = self.realtime_asr_beam_size,
                    fp16=torch.cuda.is_available(),
                    )
        return result["text"].strip()

    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.forgery_detection_mem

    def dsso_init(self,req:Dict = None)->bool:
        return True
    
    def split_string(self,
                    sentence:str,
                    pattern:str
                    ):
        result = re.findall(rf'.+?{pattern}', sentence)
        remaining_text = re.split(rf'{pattern}', sentence)[-1]
        if remaining_text:
            result.append(remaining_text)
        return result
    
    def dsso_forward(self, request):
        
        if_send = True
        if_record = False
        current_length = 0.0
        speech_timestamps = None
        result = ""
        output = {
            "text":"",
            "record":if_record,
            "if_send":if_send,
            "audio_length":current_length,
            "speech_timestamps":speech_timestamps,
            }
        #print(request)
        if request["task_id"]==None:
            return output,True
        if request["task_id"] in self.audio_tensors.keys():
            pass
        else:
            self.audio_tensors[request["task_id"]] = torch.zeros((0),dtype=torch.float)
            self.output_table[request["task_id"]] = []



        if request['state'] == "start":
            return output,False
        
        elif request['state'] == 'finished':
            output["if_send"] = True
            return output,True
        
        elif request['state'] == 'await':
            return output,True
        
        else:
            decoded_audio = base64.b64decode(request['audio_data'])
            audio_samples = np.frombuffer(decoded_audio, dtype=np.int16).astype(np.float32) / 32768.0
            # Convert the NumPy array to a PyTorch tensor
            audio_tensor = torch.from_numpy(audio_samples).float()
            #torch.Size([4096])
            
            #resample if required
            if self.realtime_asr_model_sample != request['sample_rate']:
                resampler = torchaudio.transforms.Resample(orig_freq=request['sample_rate'], new_freq=self.realtime_asr_model_sample)
                audio_tensor = resampler(audio_tensor)
            
            # concat the current segment to audio tensor table
            self.audio_tensors[request["task_id"]] = torch.cat(
                (self.audio_tensors[request["task_id"]],audio_tensor),
                dim=0
                )
            
            # total length
            current_length = self.audio_tensors[request["task_id"]].shape[0]/self.realtime_asr_model_sample



            if current_length>=self.realtime_asr_max_length_ms_chatbot/1000:
                result = self.asr_forward(self.audio_tensors[request["task_id"]],request)
                self.audio_tensors[request["task_id"]] = torch.zeros((0),dtype=torch.float)
                if_send = True
                audio_length = current_length

            elif  current_length>=self.realtime_asr_adaptive_thresholding_chatbot/1000:

                speech_timestamps = self.vad_model.predict_func_delay(
                    audio = self.audio_tensors[request["task_id"]], 
                    sampling_rate=self.realtime_asr_model_sample,
                    min_silence_duration_ms = self.realtime_asr_min_silence_duration_ms_chatbot,
                    return_seconds = True,
                    )

                if len(speech_timestamps)>0:
                    last_active_point =  speech_timestamps[-1]["end"]
                    
                    if current_length-last_active_point>= self.realtime_asr_adaptive_thresholding_chatbot/1000:

                        valid_tensor = self.audio_tensors[request["task_id"]][:]
                        self.audio_tensors[request["task_id"]] = torch.zeros((0),dtype=torch.float)

                        if valid_tensor.shape[0]==0:
                            pass
                        else:
                            result = self.asr_forward(valid_tensor,request)
                            if_send = True
                            
                else:
                    pass
            else:
                pass
            output["text"] = result
            output["if_send"] = if_send
            output["audio_length"] = current_length
            output["speech_timestamps"] = speech_timestamps
            return output,False
        

        


   