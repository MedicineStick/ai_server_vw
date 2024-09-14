from typing import Dict
import json
import base64
from vosk import Model, KaldiRecognizer
from myapp.dsso_server import DSSO_SERVER
from myapp.server_conf import ServerConfig


class Online_ASR(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        print("--->initialize Online_ASR...")
        super().__init__()
        self.conf = conf
        self._need_mem = self.conf.online_asr_mem
        self.model = None
        self.rec = None
        self.output_text = {}
        self.output_text['output'] = ['']
        self.output_text['text'] = ""
        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.forgery_detection_mem

    def dsso_init(self,req:Dict = None)->bool:
        self.conf.online_asr_sample_rate = req['sample_rate']
        if self.rec == None:
            if req['language_code'] == 'en':
                self.model = Model(self.conf.online_asr_model_en)
                self.rec = KaldiRecognizer(self.model, self.conf.online_asr_sample_rate)
            elif req['language_code'] == 'zh':
                self.model = Model(self.conf.online_asr_model_cn)
                self.rec = KaldiRecognizer(self.model, self.conf.online_asr_sample_rate)
            elif req['language_code'] == 'de':
                self.model = Model(self.conf.online_asr_model_de)
                self.rec = KaldiRecognizer(self.model, self.conf.online_asr_sample_rate)
            else:
                pass

    def dsso_forward(self, request):
        if request['state'] == "start":
            return {},False
        if request['state'] == 'finished':
            return self.output_text,True
        if request['state'] == 'reset':
            return self.rec.FinalResult(),False
        elif self.rec.AcceptWaveform(base64.b64decode(request['audio_data'])):
            temp = json.loads(self.rec.Result())
            self.output_text['output'][-1] = temp['text']
            self.output_text['output'].append('')
            self.output_text['text'] = temp['text']
            return self.output_text,False
        else:
            temp = json.loads(self.rec.PartialResult())
            self.output_text['output'][-1] = temp['partial']
            self.output_text['text'] = ""
            return self.output_text,False
