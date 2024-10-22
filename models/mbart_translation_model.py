

from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast



class MbartTranslationModel(DSSO_MODEL):
    def __init__(self, conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        self.device = torch.device(conf.gpu_id)
        self.model = MBartForConditionalGeneration.from_pretrained(conf.translation_model_path).to(self.device)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(conf.translation_model_path)


    def predict_func(self, **kwargs)->dict:
        task = kwargs["task"]
        text = kwargs["text"]
        result = ""
        if task == 'zh2en':
            # translate Hindi to French
            self.tokenizer.src_lang = "zh_CN"
            encoded_hi = self.tokenizer(text, return_tensors="pt").to(self.device)
            generated_tokens = self.model.generate(
                **encoded_hi,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"]
            )
            output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            result = output[0]
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
            result = output[0]
            # => "The Secretary-General of the United Nations says there is no military solution in Syria."
        else:
            result = 'Can not translate '+task
        return result

        