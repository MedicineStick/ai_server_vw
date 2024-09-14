from typing import Dict
import json
import base64
from vosk import Model, KaldiRecognizer
from myapp.dsso_server import DSSO_SERVER
from myapp.server_conf import ServerConfig
import subprocess
import shlex
class Online_ASR_webm(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        print("--->initialize Online_ASR_webm...")
        super().__init__()
        self.conf = conf
        self._need_mem = self.conf.online_asr_mem
        self.model = None
        self.rec = None
        self.output_text = {}
        self.output_text['output'] = ['']
        self.output_text['text'] = ""

    def process_webm_ffmpeg(self,data):
        command = 'ffmpeg -i pipe:0 -f wav -'
        process = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(input=data)
        if process.returncode != 0:
            print("Error:", stderr.decode())
        else:
            pass
        return stdout

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

        data = request['audio_data']
        if data != None:
            audio = self.process_webm_ffmpeg(base64.b64decode(data))

        if request['state'] == "start":
            return {},False
        if request['state'] == 'finished':
            return self.output_text,True
        if request['state'] == 'reset':
            return self.rec.FinalResult(),False
        elif self.rec.AcceptWaveform(audio):
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
