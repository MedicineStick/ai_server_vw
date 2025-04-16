import torch
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import urllib
from realesrgan import RealESRGANer
from diffusers.utils import load_image
from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig
import numpy as np
from PIL import Image
import os
import shutil
import glob
from basicsr.utils import imwrite

from third_party.GFPGAN.gfpgan import GFPGANer

class GFPgan_args:
    def __init__(self,
                 input: str = './third_party/GFPGAN/inputs/whole_imgs',
                 output: str = './third_party/GFPGAN/results',
                 version: str = '1.4',
                 upscale: int = 2,
                 bg_upsampler: str = 'realesrgan',
                 bg_tile: int = 400,
                 suffix: str = None,
                 only_center_face: bool = False,
                 aligned: bool = False,
                 ext: str = 'auto',
                 weight: float = 0.5):
        self.input = input
        self.output = output
        self.version = version
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler
        self.bg_tile = bg_tile
        self.suffix = suffix
        self.only_center_face = only_center_face
        self.aligned = aligned
        self.ext = ext
        self.weight = weight

class GFPGan(DSSO_MODEL):
    def __init__(self,conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        print("--->initialize GFPGan...")
        self.conf = conf
        
    def predict_func(self, **kwargs)->dict:
        output_map ={"image1":"","image2":""}
        with torch.no_grad():
            input_image = kwargs["image_url"]
            if "scale" in kwargs.keys():
                self.conf.super_resolution_outscale = int(kwargs["scale"])
            else:
                pass
            img_path = "third_party/GFPGAN/inputs/whole_imgs/"
            img_name = input_image[input_image.rfind('/') + 1:]
            saved_path = os.path.join(img_path,img_name)
            if 'http' in input_image:
                    urllib.request.urlretrieve(input_image,saved_path)
            else:
                if os.path.exists(saved_path):
                    pass
                else:
                    shutil.copy(input_image, saved_path)

            args = GFPgan_args()
            args.upscale = self.conf.super_resolution_outscale
            args.input = saved_path
            os.makedirs(args.output, exist_ok=True)

            # ------------------------ set up background upsampler ------------------------
            if args.bg_upsampler == 'realesrgan':
                if not torch.cuda.is_available():  # CPU
                    import warnings
                    warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                                'If you really want to use it, please modify the corresponding codes.')
                    bg_upsampler = None
                else:
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from realesrgan import RealESRGANer
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                    bg_upsampler = RealESRGANer(
                        scale=2,
                        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                        model=model,
                        tile=args.bg_tile,
                        tile_pad=10,
                        pre_pad=0,
                        half=True)  # need to set False in CPU mode
            else:
                bg_upsampler = None

            # ------------------------ set up GFPGAN restorer ------------------------
            if args.version == '1':
                arch = 'original'
                channel_multiplier = 1
                model_name = 'GFPGANv1'
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
            elif args.version == '1.2':
                arch = 'clean'
                channel_multiplier = 2
                model_name = 'GFPGANCleanv1-NoCE-C2'
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
            elif args.version == '1.3':
                arch = 'clean'
                channel_multiplier = 2
                model_name = 'GFPGANv1.3'
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
            elif args.version == '1.4':
                arch = 'clean'
                channel_multiplier = 2
                model_name = 'GFPGANv1.4'
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
            elif args.version == 'RestoreFormer':
                arch = 'RestoreFormer'
                channel_multiplier = 2
                model_name = 'RestoreFormer'
                url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
            else:
                raise ValueError(f'Wrong model version {args.version}.')

            # determine model paths
            model_path = os.path.join('./third_party/GFPGAN/experiments/pretrained_models', model_name + '.pth')
            if not os.path.isfile(model_path):
                model_path = os.path.join('./third_party/GFPGAN/gfpgan/weights', model_name + '.pth')
            if not os.path.isfile(model_path):
                # download pre-trained models from url
                model_path = url

            restorer = GFPGANer(
                model_path=model_path,
                upscale=args.upscale,
                arch=arch,
                channel_multiplier=channel_multiplier,
                bg_upsampler=bg_upsampler,
                device=self.conf.gpu_id)

            # ------------------------ restore ------------------------
            # read image
            img_name = os.path.basename(args.input)
            print(f'Processing {img_name} ...')
            basename, ext = os.path.splitext(img_name)
            input_img = cv2.imread(args.input, cv2.IMREAD_COLOR)

            # restore faces and background if necessary
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                input_img,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=True,
                weight=args.weight)

            # save faces
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                # save cropped face
                save_crop_path = os.path.join(args.output, 'cropped_faces', f'{basename}_{idx:02d}.png')
                imwrite(cropped_face, save_crop_path)
                # save restored face
                if args.suffix is not None:
                    save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
                else:
                    save_face_name = f'{basename}_{idx:02d}.png'
                save_restore_path = os.path.join(args.output, 'restored_faces', save_face_name)
                imwrite(restored_face, save_restore_path)
                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{idx:02d}.png'))

            # save restored img
            save_restore_path, save_restore_path_resized = "", ""
            if restored_img is not None:
                if args.ext == 'auto':
                    extension = ext[1:]
                else:
                    extension = args.ext

                if args.suffix is not None:
                    save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
                    save_restore_path_resized = os.path.join(args.output, 'restored_imgs', f'{basename}_{args.suffix}_resized.{extension}')
                else:
                    save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
                    save_restore_path_resized = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
                imwrite(restored_img, save_restore_path)
            print("save_restore_path: ",save_restore_path)

            restore_img = cv2.imread(save_restore_path, cv2.IMREAD_COLOR)

            if restore_img is None:
                raise ValueError(f"Failed to load image from {save_restore_path}")

            output_resized = cv2.resize(restore_img, (input_img.shape[1], input_img.shape[0]))

            cv2.imwrite(save_restore_path_resized, output_resized)

            output_map["image1"] = Image.open(save_restore_path_resized)
            output_map["image2"] = output_map["image1"]

            return output_map
