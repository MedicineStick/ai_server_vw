import torch
from typing import Dict
from myapp.dsso_server import DSSO_SERVER 
from myapp.server_conf import ServerConfig
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class mbart_translation(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        super().__init__()
        print("--->initialize mbart_translation...")
        self.conf = conf
        self._need_mem = conf.translation_mem
        self.device = torch.device(self.conf.gpu_id)
        self.model = MBartForConditionalGeneration.from_pretrained("myapp/resource/translation").to(self.device)
        self.tokenizer = MBart50TokenizerFast.from_pretrained("myapp/resource/translation")
        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = conf.translation_mem

    def dsso_init(self,req:Dict = None)->bool:
        pass

        
    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {}
        task = request["task"]
        text = request["text"]
        
        if task == 'zh2en':
            # translate Hindi to French
            self.tokenizer.src_lang = "zh_CN"
            encoded_hi = self.tokenizer(text, return_tensors="pt").to(self.device)
            generated_tokens = self.model.generate(
                **encoded_hi,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"]
            )
            output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            output_map['result'] = output[0]
            # => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."
        elif task == 'en2zh':
            # translate Arabic to English
            self.tokenizer.src_lang = "en_XX"
            encoded_ar = self.tokenizer(text, return_tensors="pt").to(self.device)
            generated_tokens = self.model.generate(
                **encoded_ar,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["zh_CN"]
            )
            output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            output_map['result'] = output[0]
            # => "The Secretary-General of the United Nations says there is no military solution in Syria."
        else:
            output_map['result'] = 'Can not translate '+task
            pass
        output_map['state'] = 'finished'
        return output_map,True
