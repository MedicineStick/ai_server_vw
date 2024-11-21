import sys
import torch
from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig
import base64
sys.path.append("./third_party/vits_cn/")
from third_party.vits_cn.text.symbols import symbols
from third_party.vits_cn.vits_pinyin import VITS_PinYin
from third_party.vits_cn.text import cleaned_text_to_sequence
from third_party.vits_cn import utils
sys.path.pop()


class VitsTTSCN(DSSO_MODEL):
    def __init__(self,conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        print("--->initialize VitsTTSCN...")
        self.conf = conf
        self.device = torch.device(self.conf.gpu_id)
        self.tts_front = VITS_PinYin("./third_party/vits_cn/bert", self.device)
        config='./third_party/vits_cn/configs/bert_vits.json'
        model='./checkpoints/vit/vits_bert_model.pth'
        hps = utils.get_hparams_from_file(config)
        self.net_g = utils.load_class('third_party.vits_cn.models.SynthesizerEval')(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        utils.load_model(model, self.net_g)
        self.net_g.eval()
        self.net_g.to(self.device)

    def predict_func(self, **kwargs)->dict:
        output_map = {}
        #_ = kwargs["gender"]
        item = kwargs["text"]
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
        return output_map
