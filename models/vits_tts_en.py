import sys
import torch
from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig
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

class VitsTTSEN(DSSO_MODEL):
    def __init__(self,conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        print("--->initialize VitsTTSEN...")
        self.conf = conf
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
        _ = utils.load_checkpoint("./checkpoints/resource/tts/pretrained_ljs.pth", self.net_g_female, None)

        self.hps_male = utils.get_hparams_from_file("../../softwares/vits-main/configs//vctk_base.json")
        self.net_g_male = SynthesizerTrn(
            len(symbols),
            self.hps_male.data.filter_length // 2 + 1,
            self.hps_male.train.segment_size // self.hps_male.data.hop_length,
            n_speakers=self.hps_male.data.n_speakers,
            **self.hps_male.model).cuda().to(self.device)
        _ = self.net_g_male.eval()
        _ = utils.load_checkpoint("./checkpoints/resource/tts/pretrained_vctk.pth", self.net_g_male, None)

        
    def predict_func(self, **kwargs)->dict:

        output_map = {}
        gender = kwargs["gender"]
        text = kwargs["text"]

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
        return output_map 
