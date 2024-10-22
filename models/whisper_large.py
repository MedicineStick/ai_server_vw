

from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig

import torch
import whisper

class WhisperLarge(DSSO_MODEL):
    def __init__(self, conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        device = torch.device(conf.gpu_id)
        self.asr_model = whisper.load_model(
                name=conf.ai_meeting_whisper_model_name,
                device=device,
                download_root=conf.ai_meeting_asr_model_path
                )
        
    def predict_func(self, **kwargs)->dict:
        # audio, word_timestamps, language, beam_size
        results = self.asr_model.transcribe(**kwargs)
        return results

        