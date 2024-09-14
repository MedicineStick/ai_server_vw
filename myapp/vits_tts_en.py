import sys
import torch
from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from myapp.server_conf import ServerConfig
import base64
sys.path.append("./third_party/vits_en/")
from third_party.vits_en import commons
from third_party.vits_en import utils
from third_party.vits_en.models import SynthesizerTrn
from third_party.vits_en.text.symbols import symbols
from third_party.vits_en.text import text_to_sequence
sys.path.pop()

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

class vits_tts_en(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        super().__init__()
        print("--->initialize vits_tts_en...")
        self.conf = conf
        self._need_mem = conf.tts_en_mem
        self.device = torch.device(self.conf.gpu_id)
        self.net_g_female = None
        self.net_g_male = None
        self.hps_female = utils.get_hparams_from_file("../../softwares/vits-main/configs//ljs_base.json")
        self.net_g_female = SynthesizerTrn(
            len(symbols),
            self.hps_female.data.filter_length // 2 + 1,
            self.hps_female.train.segment_size // self.hps_female.data.hop_length,
            **self.hps_female.model).cuda().to(self.device)
        _ = self.net_g_female.eval()
        _ = utils.load_checkpoint("./myapp/resource/tts/pretrained_ljs.pth", self.net_g_female, None)

        self.hps_male = utils.get_hparams_from_file("../../softwares/vits-main/configs//vctk_base.json")
        self.net_g_male = SynthesizerTrn(
            len(symbols),
            self.hps_male.data.filter_length // 2 + 1,
            self.hps_male.train.segment_size // self.hps_male.data.hop_length,
            n_speakers=self.hps_male.data.n_speakers,
            **self.hps_male.model).cuda().to(self.device)
        _ = self.net_g_male.eval()
        _ = utils.load_checkpoint("./myapp/resource/tts/pretrained_vctk.pth", self.net_g_male, None)

        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.tts_en_mem

    def dsso_init(self,req:Dict = None)->bool:
        if self.net_g_female == None:
            self.hps_female = utils.get_hparams_from_file("../../softwares/vits-main/configs//ljs_base.json")
            self.net_g_female = SynthesizerTrn(
                len(symbols),
                self.hps_female.data.filter_length // 2 + 1,
                self.hps_female.train.segment_size // self.hps_female.data.hop_length,
                **self.hps_female.model).cuda().to(self.device)
            _ = self.net_g_female.eval()
            _ = utils.load_checkpoint("./myapp/resource/tts/pretrained_ljs.pth", self.net_g_female, None)
        if self.net_g_male == None:
            self.hps_male = utils.get_hparams_from_file("../../softwares/vits-main/configs//vctk_base.json")
            self.net_g_male = SynthesizerTrn(
                len(symbols),
                self.hps_male.data.filter_length // 2 + 1,
                self.hps_male.train.segment_size // self.hps_male.data.hop_length,
                n_speakers=self.hps_male.data.n_speakers,
                **self.hps_male.model).cuda().to(self.device)
            _ = self.net_g_male.eval()
            _ = utils.load_checkpoint("./myapp/resource/tts/pretrained_vctk.pth", self.net_g_male, None)

        
    def dsso_forward(self, request: Dict) -> Dict:

        output_map = {}
        gender = request["gender"]
        text = request["text"]

        if gender ==0:
            stn_tst = get_text(text, self.hps_female)
            with torch.no_grad():
                x_tst = stn_tst.cuda().unsqueeze(0).to(self.device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda().to(self.device)
                audio = self.net_g_female.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        elif gender ==1:
            stn_tst = get_text(text, self.hps_male)
            with torch.no_grad():
                x_tst = stn_tst.cuda().unsqueeze(0).to(self.device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda().to(self.device)
                sid = torch.LongTensor([4]).cuda().to(self.device)
                audio = self.net_g_male.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

        binary_stream = audio.tobytes()
        encoded_audio = base64.b64encode(binary_stream).decode()
        output_map['audio_data'] = encoded_audio
        output_map['state'] = 'finished'
        return output_map,True
