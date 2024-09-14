import sys
import torch
from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from myapp.server_conf import ServerConfig
import base64
sys.path.append("./third_party/vits_cn/")
from third_party.vits_cn.text.symbols import symbols
from third_party.vits_cn.vits_pinyin import VITS_PinYin
from third_party.vits_cn.text import cleaned_text_to_sequence
from third_party.vits_cn import utils



class vits_tts_cn(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        super().__init__()
        print("--->initialize vits_tts_cn...")
        self.conf = conf
        self._need_mem = conf.tts_cn_mem
        self.device = torch.device(self.conf.gpu_id)
        self.tts_front = VITS_PinYin("./third_party/vits_cn/bert", self.device)
        config='./third_party/vits_cn/configs/bert_vits.json'
        model='./third_party/vits_cn/vits_bert_model.pth'
        hps = utils.get_hparams_from_file(config)
        self.net_g = utils.load_class('third_party.vits_cn.models.SynthesizerEval')(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        utils.load_model(model, self.net_g)
        self.net_g.eval()
        self.net_g.to(self.device)



        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.tts_cn_mem

    def dsso_init(self,req:Dict = None)->bool:
        pass

        
    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {}
        _ = request["gender"]
        item = request["text"]
        if (item == None or item == ""):
            return output_map,True
        phonemes, char_embeds = self.tts_front.chinese_to_phonemes(item)
        input_ids = cleaned_text_to_sequence(phonemes)
        with torch.no_grad():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(self.device)
            x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5,
                                length_scale=1)[0][0, 0].data.cpu().float().numpy()
        binary_stream = audio.tobytes()
        encoded_audio = base64.b64encode(binary_stream).decode()
        output_map['audio_data'] = encoded_audio
        output_map['state'] = 'finished'
        return output_map,True
