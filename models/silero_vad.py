

from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig

import torch
#/home/tione/notebook/lskong2/projects/ai_server_vw/third_party/silero-vad-master/utils_vad.py

class Silero_VAD(DSSO_MODEL):
    def __init__(self, conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        device = torch.device(conf.gpu_id)
        self.if_available = True
        self.conf = conf
        print("Loading VAD model...")
        self.vad_model, self.utils = torch.hub.load(repo_or_dir=self.conf.ai_meeting_vad_dir,
                                    model='silero_vad',
                                    source='local',
                                    force_reload=False,
                                    onnx=True)
        (self.get_speech_timestamps,
        _,
        self.read_audio,
        _,
        _) = self.utils

        
    def predict_func(self, **kwargs)->dict:

        audio = kwargs.pop("audio")
        if isinstance(audio,str):
            audio = self.read_audio(audio, sampling_rate=kwargs["sampling_rate"])

        speech_timestamps = self.get_speech_timestamps(
                    audio = audio,
                    model = self.vad_model,
                    **kwargs
                    )

        return speech_timestamps

        