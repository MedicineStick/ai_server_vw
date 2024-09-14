import torch
from transformers.models.mobilenet_v2 import MobileNetV2ForImageClassification
from transformers.models.mobilenet_v2 import MobileNetV2Config, MobileNetV2Model
from torchvision import transforms
from diffusers.utils import load_image
from typing import Dict
from myapp.dsso_server import DSSO_SERVER
from myapp.server_conf import ServerConfig


class AI_Classification(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        print("--->initialize AI_Classification...")
        super().__init__()
        self.conf = conf
        self._need_mem = self.conf.ai_classification_mem
        self.device = torch.device(self.conf.gpu_id)
        configuration = MobileNetV2Config()
        configuration.num_labels = self.conf.ai_classification_num_class
        configuration.problem_type = self.conf.ai_classification_problem_type
        model = MobileNetV2Model(configuration)
        configuration = model.config
        self.model = MobileNetV2ForImageClassification(
            configuration
            ).from_pretrained(
                self.conf.ai_classification_path,
                ignore_mismatched_sizes=True
                )
        self.model.to(self.device)
        self.model.training = False

    def dsso_init(self,req:Dict = None)->bool:
        pass
        
    
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = self.conf.ai_classification_mem
        self.device = torch.device(self.conf.gpu_id)


    def dsso_forward(self, request: Dict) -> Dict:
        with torch.no_grad():
            transform = transforms.Compose([
            transforms.Resize((self.conf.ai_classification_input_sz, self.conf.ai_classification_input_sz)),  # Resize the image
            transforms.ToTensor(),           # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        
            self.device = torch.device(self.conf.gpu_id)
            softmax_func = torch.nn.Softmax(dim=0)
            

            image_url = request["image_url"]
            #from diffusers.utils import load_image
            #img1 = load_image(image_url)
            #img = cv2.cvtColor(np.asarray(img1),cv2.COLOR_RGB2BGR)  
            image = load_image(image_url)
            image = image.convert('RGB')
            image = transform(image)
            image = image.to(self.device).unsqueeze(0)
            

            output = self.model(pixel_values=image,
                                    output_hidden_states=False
                        )
            output = output.logits[0].detach()
            logits = (softmax_func(output)*10000)/100
            output_map = {}
            output_map["output"] = logits.tolist()
            torch.cuda.empty_cache()
            output_map['state'] = 'finished'
            return output_map,True
