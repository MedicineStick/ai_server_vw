from typing import Dict
import numpy as np
import torch
import base64
from myapp.dsso_server import DSSO_SERVER
from models.server_conf import ServerConfig
from typing import Any
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
import concurrent.futures.thread
import asyncio
class Realtime_ASR_Whisper_Silero_Vad(DSSO_SERVER):
    def __init__(
            self,
            conf:ServerConfig,
            asr_model:DSSO_MODEL,
            vad_model:DSSO_MODEL,
            translation_model:DSSO_MODEL,
            executor:concurrent.futures.thread.ThreadPoolExecutor,
            time_blocker:int
            ):
        print("--->initialize Realtime_ASR_Whisper_Silero_Vad...")
        super().__init__(time_blocker=time_blocker)
        self.executor = executor
        self.conf = conf
        self._need_mem = self.conf.online_asr_mem
        self.model = None
        self.rec = None
        self.output_text = {}
        self.mbart_translation_model = translation_model
        self.audio_tensors:dict[str:torch.Tensor] = {}
        self.if_translation = False
        self.output_table:dict[str:dict] = {}
        self.realtime_asr_min_combine_sents_sec = self.conf.realtime_asr_min_combine_sents_sec
        self.realtime_asr_model_sample = self.conf.realtime_asr_model_sample
        self.realtime_asr_gap_ms = self.conf.realtime_asr_gap_ms
        self.realtime_asr_beam_size = self.conf.realtime_asr_beam_size
        self.realtime_asr_min_silence_duration_ms = self.conf.realtime_asr_min_silence_duration_ms
        self.vad_model = vad_model


        #/home/tione/notebook/lskong2/projects/ai_server_vw/third_party/silero-vad-master/utils_vad.py
        self.punctuation = set()
        self.punctuation.add('.')
        self.punctuation.add('!')
        self.punctuation.add('?')
        self.punctuation.add('。')
        self.punctuation.add('！')
        self.punctuation.add('？')

        self.pattern = ''.join(self.punctuation).strip()
        self.pattern1 = f'[{"".join(re.escape(p) for p in self.punctuation)}]'

        self.asr_model = asr_model

    async def asyn_forward(self, websocket,message):
        import json
        response = await asyncio.get_running_loop().run_in_executor(self.executor, self.dsso_forward, message)
        await websocket.send(json.dumps(response))

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
                        result[-1]["output"] = result[-1]["output"]+' '+last["output"]
                        result[-1]["refactoring"]=True
                        result[-1]["timestamp_end"] = last["timestamp_end"]
                    else:
                        result[-1]["output"] = result[-1]["output"]+' '+last["output"]
                        result[-1]["refactoring"]=False
                        result[-1]["timestamp_end"] = last["timestamp_end"]
            else:
                if last['output'][-1] in self.punctuation:
                    last["refactoring"]=True
                    result.append(last)
                else:
                    result.append(last)
        else:
            for i in range(0,len(split_result)):
                timestamp_chunk = (last["timestamp_end"]-last["timestamp_start"])/len(split_result)

                if i == len(split_result)-1:
                    if split_result[i][-1] in self.punctuation:
                        result.append(
                            {
                            "output":split_result[i],
                            "trans": None,
                            "refactoring":True,
                            "timestamp_start":result[-1]["timestamp_end"],
                            "timestamp_end":result[-1]["timestamp_end"]+timestamp_chunk     
                            }
                        )
                    else:
                        result.append(
                            {
                            "output":split_result[i],
                            "trans": None,
                            "refactoring":False,
                            "timestamp_start":result[-1]["timestamp_end"],
                            "timestamp_end":result[-1]["timestamp_end"]+timestamp_chunk      
                            }
                        )
                
                elif i == 0:
                    if len(result)>0:
                        if result[-1]["refactoring"]: 
                            result.append(
                                {
                                "output":split_result[i],
                                "trans": None,
                                "refactoring":True,
                                "timestamp_start":result[-1]["timestamp_end"],
                                "timestamp_end":result[-1]["timestamp_end"]+timestamp_chunk      
                                }
                            )
                        else:
                            result[-1]["output"] = result[-1]["output"]+' '+split_result[i]
                            result[-1]["refactoring"]=True
                            result[-1]["timestamp_end"] = result[-1]["timestamp_end"]+timestamp_chunk
                    else:
                        result.append(
                            {
                            "output":split_result[i],
                            "trans": None,
                            "refactoring":True, 
                            "timestamp_start":0,
                            "timestamp_end":timestamp_chunk 
                            }
                        )
                else:
                    result.append(
                            {
                            "output":split_result[i],
                            "trans": None,
                            "refactoring":True,
                            "timestamp_start":result[-1]["timestamp_end"],
                            "timestamp_end":result[-1]["timestamp_end"]+timestamp_chunk    
                            }
                        )



    def dsso_forward(self, request):
        
        if_send = False
        output = {"key":{},"if_send":if_send}
        #print(request)
        if request["task_id"]==None:
            return output

        if request["task_id"] in self.audio_tensors.keys():
            pass
        else:
            self.audio_tensors[request["task_id"]] = torch.zeros((0),dtype=torch.float)
            self.output_table[request["task_id"]] = []

        if request['state'] == "start":
            return {"key":self.output_table[request["task_id"]],"if_send":if_send}
        elif request['state'] == 'finished':
            #temp = self.output_table[request["task_id"]]
            #self.output_table.pop(request["task_id"])

            return {"key":self.output_table[request["task_id"]],"if_send":True}
        else:
            decoded_audio = base64.b64decode(request['audio_data'])

            audio_samples = np.frombuffer(decoded_audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Convert the NumPy array to a PyTorch tensor
            audio_tensor = torch.from_numpy(audio_samples).float()
            #torch.Size([4096])
            
            if self.realtime_asr_model_sample != request['sample_rate']:
                    resampler = torchaudio.transforms.Resample(orig_freq=request['sample_rate'], new_freq=self.realtime_asr_model_sample)
                    audio_tensor = resampler(audio_tensor)
            
            self.audio_tensors[request["task_id"]] = torch.cat(
                (self.audio_tensors[request["task_id"]],audio_tensor),
                dim=0
                )
            
            current_length = self.audio_tensors[request["task_id"]].shape[0]/self.realtime_asr_model_sample

            if current_length>=self.realtime_asr_min_combine_sents_sec:
                
                valid_tensor = torch.zeros((0),dtype=torch.float)
                
                speech_timestamps = self.vad_model.predict_func_delay(
                    audio = self.audio_tensors[request["task_id"]], 
                    sampling_rate=self.realtime_asr_model_sample,
                    min_silence_duration_ms = self.realtime_asr_min_silence_duration_ms,
                    )
                speech_timestamps_list = process_timestamps(speech_timestamps)
                if len(speech_timestamps_list)>0:   
                    if speech_timestamps_list[-1][1]/self.realtime_asr_model_sample+self.realtime_asr_gap_ms >= self.audio_tensors[request["task_id"]].shape[0]/self.realtime_asr_model_sample:
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
                    if result == None:
                        pass
                    else:
                        text_ = result['text'].strip()
                        print(text_)
                        if len(text_)>0:

                            last_end = 0.
                            if len(self.output_table[request["task_id"]])==0:
                                pass
                            else:
                                last_end = self.output_table[request["task_id"]][-1]["timestamp_end"]

                            self.output_table[request["task_id"]].append(
                                {
                                "output":text_,
                                "trans":None,
                                "refactoring":False,
                                "timestamp_start":last_end,
                                "timestamp_end":last_end+valid_tensor.shape[0]/self.realtime_asr_model_sample
                                }
                                )
                            print(last_end,last_end+valid_tensor.shape[0]/self.realtime_asr_model_sample)
                            self.refactoring_result(self.output_table[request["task_id"]])
                            if request['translation_task'] ==1:
                                self._translation_callback(self.output_table[request["task_id"]],request['language_code'])
                            if_send = True
                else:
                    pass
        

        return {"key":self.output_table[request["task_id"]],"if_send":if_send}


    def _translation_callback(
        self,  
        result:list[dict[str:Any]],
        language_code:str
        ):

        for i in range(len(result) - 1, -1, -1):
            if result[i]["trans"]==None:
                if result[i]["refactoring"]:
                    if language_code=="en":

                        output_trans = self.mbart_translation_model.predict_func_delay(
                            task = 'en2zh',
                            text = result[i]["output"]
                        )
                        result[i]["trans"] = output_trans.strip()
                    elif language_code=="zh":

                        output_trans = self.mbart_translation_model.predict_func_delay(
                            task = 'zh2en',
                            text = result[i]["output"]
                        )
                        result[i]["trans"] = output_trans.strip()
                    else:
                        continue
                else:
                    continue
            else:
                continue
