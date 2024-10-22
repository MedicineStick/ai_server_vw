

from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig
from typing import Any

import torch
from transformers.models.mobilenet_v2 import MobileNetV2ForImageClassification
from transformers.models.mobilenet_v2 import MobileNetV2Config, MobileNetV2Model
from torchvision import transforms
from diffusers.utils import load_image

class AiClassificationModel(DSSO_MODEL):
    def __init__(self, conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        self.model = None
        
        self.transform = transforms.Compose([
            transforms.Resize((conf.ai_classification_input_sz,conf.ai_classification_input_sz)),  # Resize the image
            transforms.ToTensor(),           # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        self.softmax_func = torch.nn.Softmax(dim=0)


        self.device = torch.device(conf.gpu_id)
        configuration = MobileNetV2Config()
        configuration.num_labels = conf.ai_classification_num_class
        configuration.problem_type = conf.ai_classification_problem_type
        model = MobileNetV2Model(configuration)
        configuration = model.config
        self.model = MobileNetV2ForImageClassification(
            configuration
            ).from_pretrained(
                conf.ai_classification_path,
                ignore_mismatched_sizes=True
                )
        self.model.to(self.device)
        self.model.training = False

    def predict_func(self, **kwargs)->dict:
        output = {"output":None}

        image_url = kwargs["image_url"]
        image = load_image(image_url)
        image = image.convert('RGB')
        image = self.transform(image)
        image = image.to(self.device).unsqueeze(0)
        

        output = self.model(pixel_values=image,
                                output_hidden_states=False
                    )
        output = output.logits[0].detach()
        logits = (self.softmax_func(output)*10000)/100
        return logits.tolist()

