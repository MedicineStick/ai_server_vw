from typing import Dict
import numpy as np
import torch
import base64
from myapp.dsso_server import DSSO_SERVER
from myapp.server_conf import ServerConfig
from myapp.mbart_translation import mbart_translation
# https://github.com/davabase/whisper_real_time/tree/master
import sys
sys.path.append("/home/tione/notebook/lskong2/projects/2.tingjian/")
import whisper
import torchaudio


class Realtime_ASR_Whisper(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        print("--->initialize Online_ASR...")
        super().__init__()
        self.conf = conf
        self._need_mem = self.conf.online_asr_mem
        self.model = None
        self.rec = None
        self.output_text = {}
        self.mbart_translation_model = mbart_translation(conf)
        self.audio_tensors:dict[str:list] = {}
        self.output_table:dict[str:dict] = {}

        self.if_translation = False

        self.asr_model = whisper.load_model(
                name=self.conf.realtime_asr_whisper_model_name,
                download_root=self.conf.ai_meeting_asr_model_path
                )
        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.forgery_detection_mem

    def dsso_init(self,req:Dict = None)->bool:
        return True

    def dsso_forward(self, request):
        output = {"text":""}
        if request["task_id"]==None:
            return output,False

        if request["task_id"] in self.audio_tensors.keys():
            pass
        else:
            self.audio_tensors[request["task_id"]] = []
            self.output_table[request["task_id"]] = {"output":[]} 

        if request['state'] == "start":
            return self.output_table[request["task_id"]],False
        elif request['state'] == 'finished':
            return self.output_table[request["task_id"]],True
        else:
            decoded_audio = base64.b64decode(request['audio_data'])

            audio_samples = np.frombuffer(decoded_audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Convert the NumPy array to a PyTorch tensor
            audio_tensor = torch.from_numpy(audio_samples).float()
            #print(audio_tensor.shape)
            self.audio_tensors[request["task_id"]].append(audio_tensor)
            if len(self.audio_tensors[request["task_id"]])>=6:
                full_audio_tensor = torch.cat(self.audio_tensors[request["task_id"]], dim=0)
                resampler = torchaudio.transforms.Resample(orig_freq=request['sample_rate'], new_freq=16000)
                waveform = resampler(full_audio_tensor)
                result = self.asr_model.transcribe(waveform, language=request['language_code'], fp16=torch.cuda.is_available())
                text_ = result['text'].strip()

                if len(text_)>0:
                    self.output_table[request["task_id"]]["output"].append(text_)
                    if self.if_translation:
                        if request['language_code']=="en":
                            request_trans = {}
                            request_trans["task"] = 'en2zh'
                            request_trans["text"] = text_
                            output_trans,_ = self.mbart_translation_model.dsso_forward(request_trans)
                            self.output_table[request["task_id"]]["output"].append(output_trans['result'].strip())
                        elif request['language_code']=="zh":
                            request_trans = {}
                            request_trans["task"] = 'zh2en'
                            request_trans["text"] = text_
                            output_trans,_ = self.mbart_translation_model.dsso_forward(request_trans)
                            self.output_table[request["task_id"]]["output"].append(output_trans['result'].strip())
                        else:
                            pass
                self.audio_tensors[request["task_id"]].clear()
            else:
                pass
        return self.output_table[request["task_id"]],False
