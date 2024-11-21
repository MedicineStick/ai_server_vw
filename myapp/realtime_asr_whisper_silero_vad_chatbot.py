from typing import Dict
import numpy as np
import torch
import base64
from myapp.dsso_server import DSSO_SERVER
from models.server_conf import ServerConfig
import torchaudio
# https://github.com/snakers4/silero-vad/blob/master/examples/cpp/silero-vad-onnx.cpp
# https://github.com/mozilla/DeepSpeech-examples/blob/r0.8/mic_vad_streaming/mic_vad_streaming.py
# https://github.com/snakers4/silero-vad/blob/master/examples/microphone_and_webRTC_integration/microphone_and_webRTC_integration.py
# https://github.com/davabase/whisper_real_time/tree/master
import sys
sys.path.append("/home/tione/notebook/lskong2/projects/2.tingjian/")
from models.dsso_model import DSSO_MODEL
import torchaudio
import concurrent.futures.thread
import asyncio
class Realtime_ASR_Whisper_Silero_Vad_Chatbot(DSSO_SERVER):
    def __init__(
            self,
            conf:ServerConfig,
            asr_model:DSSO_MODEL,
            vad_model:DSSO_MODEL,
            llm_model:DSSO_MODEL,
            cn_tts_model:DSSO_MODEL,
            en_tts_model:DSSO_MODEL,
            executor:concurrent.futures.thread.ThreadPoolExecutor,
            time_blocker:int
            ):
        print("--->initialize Realtime_ASR_Whisper_Silero_Vad_Chatbot...")
        super().__init__(time_blocker=time_blocker)
        self.executor = executor
        self.conf = conf
        self._need_mem = self.conf.online_asr_mem
        self.model = None
        self.task_tables:dict[str:dict] = {}
        self.output_table:dict[str:dict] = {}
        self.realtime_asr_model_sample = self.conf.realtime_asr_model_sample
        self.realtime_asr_gap_ms = self.conf.realtime_asr_gap_ms
        self.realtime_asr_beam_size = self.conf.realtime_asr_beam_size
        self.realtime_asr_min_silence_duration_ms_chatbot = self.conf.realtime_asr_min_silence_duration_ms_chatbot
        self.realtime_asr_max_length_ms_chatbot = self.conf.realtime_asr_max_length_ms_chatbot
        self.realtime_asr_adaptive_thresholding_chatbot = self.conf.realtime_asr_adaptive_thresholding_chatbot
        self.vad_model = vad_model
        self.asr_model = asr_model
        self.llm_model = llm_model
        self.cn_tts_model = cn_tts_model
        self.en_tts_model = en_tts_model
        self.realtime_asr_llm_timeout = self.conf.realtime_asr_llm_timeout
        self.realtime_asr_retry_count = self.conf.realtime_asr_retry_count
        self.hallucination_words = ["明镜"]
    

    async def asyn_forward(self, websocket,message):
        import json
        r1 = await asyncio.get_running_loop().run_in_executor(self.executor, self.dsso_forward, message)
            
        if r1['if_wait']:
            await websocket.send(json.dumps(r1))
            r2 = await asyncio.get_running_loop().run_in_executor(self.executor, self.__execute_task, message)
            r3 = {**r1, **r2}
            r3['if_wait'] = False
            await websocket.send(json.dumps(r3))

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
    
    def __execute_task(
            self,
            request:dict,
            ):
        result = {}
        result["trans_text"] = ""
        result["response_text"] = ""
        trans_text = self.asr_forward(self.task_tables[request["task_id"]]["audio"],request)
        self.task_tables[request["task_id"]]["audio"] = torch.zeros((0),dtype=torch.float)
        
        
        if_discard = False

        for word in self.hallucination_words:
            if word in response_text:
                if_discard = True
                break
        if if_discard:
            result["response_text"] = ""
            result["trans_text"] = ""
        else:
            response_text = self.llm_model.predict_func_delay(
                            prompt = trans_text,
                            retry_count = self.realtime_asr_retry_count,
                            timeout = self.realtime_asr_llm_timeout,
                            count = 0,
                            )
            result["response_text"] = response_text
            result["trans_text"] = trans_text
        
        print("trans_text: "+result["trans_text"])
        print("response_text: "+result["response_text"])
        """
        tts_input = {}
        tts_input["text"] = result["response_text"] 
        result["audio_data"] = self.cn_tts_model.predict_func_delay(tts_input)
        """
        return result

    def dsso_forward(self, request):

        current_length = 0.0
        speech_timestamps = None
        if_wait = False
        response_text = ""
        trans_text = ""
        output = {
            "trans_text":trans_text,
            "response_text":response_text,
            "audio_length":current_length,
            "speech_timestamps":speech_timestamps,
            "if_wait":if_wait,
            }
        #print(request)
        if request["task_id"]==None:
            return output
        if request["task_id"] in self.task_tables.keys():
            pass
        else:
            self.task_tables[request["task_id"]] = {
                "audio":torch.zeros((0),dtype=torch.float),
            }
            self.output_table[request["task_id"]] = []


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
        self.task_tables[request["task_id"]]["audio"] = torch.cat(
            (self.task_tables[request["task_id"]]["audio"],audio_tensor),
            dim=0
            )
        
        # total length
        current_length = self.task_tables[request["task_id"]]["audio"].shape[0]/self.realtime_asr_model_sample



        if current_length>=self.realtime_asr_max_length_ms_chatbot/1000:
            if_wait = True

        elif current_length>=self.realtime_asr_adaptive_thresholding_chatbot/1000:

            speech_timestamps = self.vad_model.predict_func_delay(
                audio = self.task_tables[request["task_id"]]["audio"], 
                sampling_rate=self.realtime_asr_model_sample,
                min_silence_duration_ms = self.realtime_asr_min_silence_duration_ms_chatbot,
                return_seconds = True,
                )

            if len(speech_timestamps)>0:
                last_active_point =  speech_timestamps[-1]["end"]
                
                if current_length-last_active_point>= self.realtime_asr_adaptive_thresholding_chatbot/1000:

                    valid_tensor = self.task_tables[request["task_id"]]["audio"][:]
                    self.task_tables[request["task_id"]]["audio"] = torch.zeros((0),dtype=torch.float)

                    if valid_tensor.shape[0]==0:
                        pass
                    else:
                        if_wait = True             
            else:
                pass
        else:
            pass

        output["trans_text"] = trans_text
        output["response_text"] = response_text
        output["audio_length"] = current_length
        output["speech_timestamps"] = speech_timestamps
        output["if_wait"] = if_wait
        return output
        

        


   