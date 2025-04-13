import torch
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import torch.onnx
from realesrgan import RealESRGANer
from diffusers.utils import load_image
from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig
import numpy as np
from PIL import Image



class ESRGan(DSSO_MODEL):
    def __init__(self,conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        print("--->initialize ESRGan...")
        self.conf = conf
        
    def predict_func(self, **kwargs)->dict:
        output_map ={"image1":"","image2":""}
        with torch.no_grad():
            image_url = kwargs["image_url"]
            dni_weight = None
            model_path = self.conf.super_resolution_path
            netscale = self.conf.super_resolution_netscale
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            # restorer
            tile_pad = self.conf.super_resolution_tile_pad
            pre_pad = self.conf.super_resolution_pre_pad
            fp32 = False
            face_enhance = False
            gpu_id = self.conf.gpu_id
            outscale=self.conf.super_resolution_outscale
            tile = self.conf.super_resolution_tile
            upsampler = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                dni_weight=dni_weight,
                model=model,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                half=not fp32,
                gpu_id=gpu_id)
            source=load_image(image_url)
            H,W = source.height, source.width
            img_np = np.array(source)
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            try:
                output, _ = upsampler.enhance(img, outscale=outscale)
                with torch.no_grad():
                        x = torch.randn((1, 3, 64, 64),dtype=torch.float16)
                        x =x.to(device=gpu_id)              
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                output_resized = cv2.resize(output,(W,H))
                output_rgb = cv2.cvtColor(output_resized, cv2.COLOR_BGR2RGB)
                pillow_image1 = Image.fromarray(output_rgb)
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                pillow_image2 = Image.fromarray(output_rgb)
                output_map["image1"] = pillow_image1
                output_map["image2"] = pillow_image2
            torch.cuda.empty_cache()
            return output_map
