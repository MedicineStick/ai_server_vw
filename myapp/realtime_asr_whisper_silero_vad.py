from typing import Dict
import numpy as np
import torch
import base64
from myapp.dsso_server import DSSO_SERVER
from myapp.server_conf import ServerConfig
from myapp.mbart_translation import mbart_translation
from typing import Any
from myapp.dsso_util import process_timestamps
import torchaudio
# https://github.com/snakers4/silero-vad/blob/master/examples/cpp/silero-vad-onnx.cpp
# https://github.com/mozilla/DeepSpeech-examples/blob/r0.8/mic_vad_streaming/mic_vad_streaming.py
# https://github.com/snakers4/silero-vad/blob/master/examples/microphone_and_webRTC_integration/microphone_and_webRTC_integration.py
# https://github.com/davabase/whisper_real_time/tree/master
import sys
sys.path.append("/home/tione/notebook/lskong2/projects/2.tingjian/")
import whisper
import torchaudio
import re

class Realtime_ASR_Whisper_Silero_Vad(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        print("--->initialize Online_ASR...")
        super().__init__()
        self.conf = conf
        self._need_mem = self.conf.online_asr_mem
        self.model = None
        self.rec = None
        self.output_text = {}
        self.mbart_translation_model = mbart_translation(conf)
        self.audio_tensors:dict[str:torch.Tensor] = {}
        self.if_translation = False
        self.output_table:dict[str:dict] = {}
        self.realtime_asr_min_combine_sents_sec = self.conf.realtime_asr_min_combine_sents_sec
        self.realtime_asr_model_sample = self.conf.realtime_asr_model_sample
        self.realtime_asr_gap_ms = self.conf.realtime_asr_gap_ms

        print("Loading VAD model...")
        model, utils = torch.hub.load(repo_or_dir=self.conf.ai_meeting_vad_dir,
                                    model='silero_vad',
                                    source='local',
                                    force_reload=False,
                                    onnx=True)

        (get_speech_timestamps,
        _,
        read_audio,
        _,
        _) = utils
        self.vad_model = model
        self.get_speech_timestamps = get_speech_timestamps
        self.read_audio = read_audio
        self.punctuation = set()
        self.punctuation.add('.')
        self.punctuation.add('!')
        self.punctuation.add('?')
        self.punctuation.add('。')
        self.punctuation.add('！')
        self.punctuation.add('？')

        self.pattern = ''.join(self.punctuation).strip()
        self.pattern1 = f'[{"".join(re.escape(p) for p in self.punctuation)}]'

        self.asr_model = whisper.load_model(
                name=self.conf.realtime_asr_whisper_model_name,
                download_root=self.conf.ai_meeting_asr_model_path
                )
        
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

    def refactoring_result(
        self,  
        result:list[dict[str:Any]]
        ):
        last = result.pop()
        second_last = None
        split_result = self.split_string(last['output'],self.pattern1)

        if len(split_result)==1:
            if len(result)>0:
                if result[-1]["refactoring"]:
                    if last['output'][-1] in self.punctuation:
                        last["refactoring"] = True
                        result.append(last)
                    else:
                        result.append(last)
                else:
                    if last['output'][-1] in self.punctuation:
                        result[-1]["output"] = second_last["output"]+' '+last["output"]
                        result[-1]["refactoring"]=True
                    else:
                        result[-1]["output"] = second_last["output"]+' '+last["output"]
                        result[-1]["refactoring"]=False
            else:
                if last['output'][-1] in self.punctuation:
                    last["refactoring"]=True
                    result.append(last)
                else:
                    result.append(last)
        else:
            for i in range(0,len(split_result)):
                if i == len(split_result)-1:
                    if split_result[i][-1] in self.punctuation:
                        result.append(
                            {
                            "output":split_result[i],
                            "trans": None,
                            "refactoring":True   
                            }
                        )
                    else:
                        result.append(
                            {
                            "output":split_result[i],
                            "trans": "",
                            "refactoring":False   
                            }
                        )
                
                elif i == 0:
                    if len(result)>0:
                        if result[-1]["refactoring"]: 
                            result.append(
                                {
                                "output":split_result[i],
                                "trans": None,
                                "refactoring":True   
                                }
                            )
                        else:
                            result[-1]["output"] = result[-1]["output"]+' '+split_result[i]
                            result[-1]["refactoring"]=True
                    else:
                        result.append(
                            {
                            "output":split_result[i],
                            "trans": None,
                            "refactoring":True   
                            }
                        )
                else:
                    result.append(
                            {
                            "output":split_result[i],
                            "trans": None,
                            "refactoring":True   
                            }
                        )



    def dsso_forward(self, request):
        output = {"text":""}
        #print(request)
        if request["task_id"]==None:
            return output,False

        if request["task_id"] in self.audio_tensors.keys():
            pass
        else:
            self.audio_tensors[request["task_id"]] = torch.zeros((0),dtype=torch.float)
            self.output_table[request["task_id"]] = []

        if request['state'] == "start":
            return {request["task_id"]:self.output_table[request["task_id"]]},False
        elif request['state'] == 'finished':

            temp = self.output_table[request["task_id"]]
            self.output_table.pop(request["task_id"])
            return {request["task_id"]:temp},True
        else:
            decoded_audio = base64.b64decode(request['audio_data'])

            audio_samples = np.frombuffer(decoded_audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Convert the NumPy array to a PyTorch tensor
            audio_tensor = torch.from_numpy(audio_samples).float()
            #torch.Size([4096])
            
            if self.realtime_asr_model_sample == request['sample_rate']:
                    resampler = torchaudio.transforms.Resample(orig_freq=request['sample_rate'], new_freq=self.realtime_asr_model_sample)
                    audio_tensor = resampler(audio_tensor)
            
            self.audio_tensors[request["task_id"]] = torch.cat(
                (self.audio_tensors[request["task_id"]],audio_tensor),
                dim=0
                )
            
            current_length = self.audio_tensors[request["task_id"]].shape[0]/request['sample_rate']

            if current_length>=self.realtime_asr_min_combine_sents_sec:
                
                valid_tensor = torch.zeros((0),dtype=torch.float)
                
                speech_timestamps = self.get_speech_timestamps(
                    self.audio_tensors[request["task_id"]], 
                    self.vad_model, 
                    sampling_rate=self.realtime_asr_model_sample,
                    min_silence_duration_ms = 500,
                    )
                speech_timestamps_list = process_timestamps(speech_timestamps)
                if len(speech_timestamps_list)>0:   
                    if speech_timestamps_list[-1][1]/request['sample_rate']+self.realtime_asr_gap_ms >= self.audio_tensors[request["task_id"]].shape[0]/request['sample_rate']:
                        speech_timestamps_list.pop()
                        if len(speech_timestamps_list)>0:
                           
                           temp_tensor = self.audio_tensors[request["task_id"]][speech_timestamps_list[-1][1]:].clone()
                           
                           valid_tensor = self.audio_tensors[request["task_id"]][:speech_timestamps_list[-1][1]].clone()

                           self.audio_tensors[request["task_id"]] = temp_tensor
                        else:
                           pass

                    else:

                        valid_tensor = self.audio_tensors[request["task_id"]][:speech_timestamps_list[-1][1]]

                        self.audio_tensors[request["task_id"]] = self.audio_tensors[request["task_id"]][speech_timestamps_list[-1][1]:].clone()
                    result = None
                    if valid_tensor.shape[0]==0:
                        pass
                    else:
                        result = self.asr_model.transcribe(
                                valid_tensor, 
                                language=request['language_code'],
                                beam_size = 8,
                                fp16=torch.cuda.is_available()
                                )
                    if result == None:
                        pass
                    else:
                        text_ = result['text'].strip()
                        if len(text_)>0: 
                            self.output_table[request["task_id"]].append(
                                {
                                "output":text_,
                                "trans":None,
                                "refactoring":False
                                }
                                )
                            self.refactoring_result(self.output_table[request["task_id"]])
                            self._translation_callback(self.output_table[request["task_id"]],request['translation_task'])
                else:
                    pass
        return {"key":self.output_table[request["task_id"]]},False

    def _translation_callback(
        self,  
        result:list[dict[str:Any]],
        task:str
        ):

        for i in range(len(result) - 1, -1, -1):
            if result[i]["trans"]==None:
                if result[i]["refactoring"]:
                    if task=="en2zh":
                        request_trans = {}
                        request_trans["task"] = 'en2zh'
                        request_trans["text"] = result[i]["output"]
                        output_trans,_ = self.mbart_translation_model.dsso_forward(request_trans)
                        result[i]["trans"] = output_trans['result'].strip()
                    elif task=="zh2en":
                        request_trans = {}
                        request_trans["task"] = 'zh2en'
                        request_trans["text"] = result[i]["output"]
                        output_trans,_ = self.mbart_translation_model.dsso_forward(request_trans)
                        result[i]["trans"] = output_trans['result'].strip()
                    else:
                        pass
                else:
                    pass
            else:
                pass
