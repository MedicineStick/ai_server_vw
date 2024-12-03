import argparse
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import sys,os
sys.path.append("./third_party/")
from motionclone.models.unet import UNet3DConditionModel
from motionclone.models.sparse_controlnet import SparseControlNetModel
from motionclone.pipelines.pipeline_animation import AnimationPipeline
from motionclone.utils.util import load_weights, auto_download, set_all_seed
from diffusers.utils.import_utils import is_xformers_available
from motionclone.utils.motionclone_functions import *
from motionclone.utils.xformer_attention import *
from typing import Dict
from models.dsso_util import CosUploader
from myapp.dsso_server import DSSO_SERVER
from models.server_conf import ServerConfig

import concurrent.futures.thread
import asyncio
# python i2v_video_sample.py --inference_config "configs/i2v_rgb.yaml" --examples "configs/i2v_rgb.jsonl"


class ARGS():
    def __init__(self) -> None:
        self.pretrained_model_path = "./checkpoints/motionclone_models/StableDiffusion"
        self.inference_config = "./checkpoints/motionclone_models/configs/i2v_rgb.yaml"
        self.examples = "./checkpoints/motionclone_models/configs/i2v_rgb.jsonl"
        self.motion_representation_save_dir = "./temp/motion_clone/motion_representation/"
        self.generated_videos_save_dir = "./temp/motion_clone/generated_videos/"
        self.visible_gpu = None
        self.default_seed = 76739
        self.L = 16
        self.H = 512
        self.W = 512
        self.without_xformers = True

def motion_exec(request: Dict) -> str:
        example_infor = request
        #example_infor = {"video_path":"temp/motion_clone/reference_videos/camera_zoom_out.mp4", "condition_image_paths":["temp/motion_clone/condition_images/rgb/dog_on_grass.png"], "new_prompt": "Dog, lying on the grass"}
        args = ARGS()
        device_id = request["motion_clone_device"]
        config  = OmegaConf.load(args.inference_config)
        adopted_dtype = torch.float16
        device = torch.device(device_id)
        set_all_seed(42)
        
        tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device).to(dtype=adopted_dtype)
        vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device).to(dtype=adopted_dtype)
        
        config.width = config.get("W", args.W)
        config.height = config.get("H", args.H)
        config.video_length = config.get("L", args.L)
        
        if not os.path.exists(args.generated_videos_save_dir):
            os.makedirs(args.generated_videos_save_dir)
        OmegaConf.save(config, os.path.join(args.generated_videos_save_dir,"inference_config.json"))
        
        model_config = OmegaConf.load("./checkpoints/motionclone_models/configs/model_config/model_config.yaml")
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(model_config.unet_additional_kwargs),).to(device).to(dtype=adopted_dtype)
        
        # load controlnet model
        controlnet =  None
        if config.get("controlnet_path", "") != "":
            # assert model_config.get("controlnet_images", "") != ""
            assert config.get("controlnet_config", "") != ""
            
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load("/home/tione/notebook/lskong2/projects/ai_server_vw/checkpoints/motionclone_models/configs/sparsectrl/latent_condition.yaml")
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {})).to(device).to(dtype=adopted_dtype)

            auto_download(config.controlnet_path, is_dreambooth_lora=False)
            print(f"loading controlnet checkpoint from {config.controlnet_path} ...")
            controlnet_state_dict = torch.load(config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict = {name: param for name, param in controlnet_state_dict.items() if "pos_encoder.pe" not in name}
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            del controlnet_state_dict

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(model_config.noise_scheduler_kwargs)),
        ).to(device)
        
        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = config.get("motion_module", ""),
            # domain adapter
            adapter_lora_path          = config.get("adapter_lora_path", ""),
            adapter_lora_scale         = config.get("adapter_lora_scale", 1.0),
            # image layer
            #dreambooth_model_path      = config.get("dreambooth_path", ""),
        ).to(device)
        pipeline.text_encoder.to(dtype=adopted_dtype)
        
        # customized functions in motionclone_functions
        pipeline.scheduler.customized_step = schedule_customized_step.__get__(pipeline.scheduler)
        pipeline.scheduler.customized_set_timesteps = schedule_set_timesteps.__get__(pipeline.scheduler)
        pipeline.unet.forward = unet_customized_forward.__get__(pipeline.unet)
        pipeline.sample_video = sample_video.__get__(pipeline)
        pipeline.single_step_video = single_step_video.__get__(pipeline)
        pipeline.get_temp_attn_prob = get_temp_attn_prob.__get__(pipeline)
        pipeline.add_noise = add_noise.__get__(pipeline)
        pipeline.compute_temp_loss = compute_temp_loss.__get__(pipeline)
        pipeline.obtain_motion_representation = obtain_motion_representation.__get__(pipeline)
        
        for param in pipeline.unet.parameters():
            param.requires_grad = False
        for param in pipeline.controlnet.parameters():
            param.requires_grad = False
        
        pipeline.input_config,  pipeline.unet.input_config = config,  config
        pipeline.unet = prep_unet_attention(pipeline.unet,pipeline.input_config.motion_guidance_blocks)
        pipeline.unet = prep_unet_conv(pipeline.unet)
        pipeline.scheduler.customized_set_timesteps(config.inference_steps, config.guidance_steps,config.guidance_scale,device=device,timestep_spacing_type = "uneven")

        config.video_path = example_infor["video_path"]
        config.condition_image_path_list = example_infor["condition_image_paths"]
        config.image_index = example_infor.get("image_index",[0])
        assert len(config.image_index) == len(config.condition_image_path_list)
        config.new_prompt = example_infor["new_prompt"] + config.get("positive_prompt", "")
        config.controlnet_scale = example_infor.get("controlnet_scale", 1.0)
        pipeline.input_config,  pipeline.unet.input_config = config,  config  # update config
        
        #  perform motion representation extraction
        seed_motion = example_infor.get("seed", args.default_seed) 
        generator = torch.Generator(device=pipeline.device)
        generator.manual_seed(seed_motion)
        if not os.path.exists(args.motion_representation_save_dir):
            os.makedirs(args.motion_representation_save_dir)
        motion_representation_path = os.path.join(args.motion_representation_save_dir,  os.path.splitext(os.path.basename(config.video_path))[0] + '.pt') 
        pipeline.obtain_motion_representation(generator= generator, motion_representation_path = motion_representation_path, use_controlnet=True,) 
        
        # perform video generation
        seed = seed_motion # can assign other seed here
        generator = torch.Generator(device=pipeline.device)
        generator.manual_seed(seed)
        pipeline.input_config.seed = seed
        videos = pipeline.sample_video(generator = generator, add_controlnet=True,)

        videos = rearrange(videos, "b c f h w -> b f h w c")
        save_path = os.path.join(args.generated_videos_save_dir, os.path.splitext(os.path.basename(config.video_path))[0]
                                    + "_" + config.new_prompt.strip().replace(' ', '_') + str(seed_motion) + "_" +str(seed)+'.mp4')                                        
        videos_uint8 = (videos[0] * 255).astype(np.uint8)
        imageio.mimwrite(save_path, videos_uint8, fps=8)
        print(save_path)
        return save_path



class Motion_Clone(DSSO_SERVER):
    
    def __init__(self,
                 conf:ServerConfig,
                 uploader:CosUploader,
                 executor:concurrent.futures.thread.ThreadPoolExecutor,
                 time_blocker:int
                 ):
        print("--->initialize AI_Classification...")
        super().__init__(time_blocker=time_blocker)
        self.uploader = uploader
        self.conf  = conf
        self.executor = executor
        self.motion_clone_device = conf.motion_clone_device
        self.args = ARGS()


        args = ARGS()
        device_id = self.conf.motion_clone_device
        config  = OmegaConf.load(args.inference_config)
        adopted_dtype = torch.float16
        device = torch.device(device_id)
        set_all_seed(42)
        
        tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device).to(dtype=adopted_dtype)
        vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device).to(dtype=adopted_dtype)
        
        config.width = config.get("W", args.W)
        config.height = config.get("H", args.H)
        config.video_length = config.get("L", args.L)
        
        if not os.path.exists(args.generated_videos_save_dir):
            os.makedirs(args.generated_videos_save_dir)
        OmegaConf.save(config, os.path.join(args.generated_videos_save_dir,"inference_config.json"))
        
        model_config = OmegaConf.load("./checkpoints/motionclone_models/configs/model_config/model_config.yaml")
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(model_config.unet_additional_kwargs),).to(device).to(dtype=adopted_dtype)
        
        # load controlnet model
        controlnet =  None
        if config.get("controlnet_path", "") != "":
            # assert model_config.get("controlnet_images", "") != ""
            assert config.get("controlnet_config", "") != ""
            
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load("/home/tione/notebook/lskong2/projects/ai_server_vw/checkpoints/motionclone_models/configs/sparsectrl/latent_condition.yaml")
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {})).to(device).to(dtype=adopted_dtype)

            auto_download(config.controlnet_path, is_dreambooth_lora=False)
            print(f"loading controlnet checkpoint from {config.controlnet_path} ...")
            controlnet_state_dict = torch.load(config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict = {name: param for name, param in controlnet_state_dict.items() if "pos_encoder.pe" not in name}
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            del controlnet_state_dict

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()

        self.pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(model_config.noise_scheduler_kwargs)),
        ).to(device)
        
        self.pipeline = load_weights(
            self.pipeline,
            # motion module
            motion_module_path         = config.get("motion_module", ""),
            # domain adapter
            adapter_lora_path          = config.get("adapter_lora_path", ""),
            adapter_lora_scale         = config.get("adapter_lora_scale", 1.0),
            # image layer
            #dreambooth_model_path      = config.get("dreambooth_path", ""),
        ).to(device)
        self.pipeline.text_encoder.to(dtype=adopted_dtype)
        
        # customized functions in motionclone_functions
        self.pipeline.scheduler.customized_step = schedule_customized_step.__get__(self.pipeline.scheduler)
        self.pipeline.scheduler.customized_set_timesteps = schedule_set_timesteps.__get__(self.pipeline.scheduler)
        self.pipeline.unet.forward = unet_customized_forward.__get__(self.pipeline.unet)
        self.pipeline.sample_video = sample_video.__get__(self.pipeline)
        self.pipeline.single_step_video = single_step_video.__get__(self.pipeline)
        self.pipeline.get_temp_attn_prob = get_temp_attn_prob.__get__(self.pipeline)
        self.pipeline.add_noise = add_noise.__get__(self.pipeline)
        self.pipeline.compute_temp_loss = compute_temp_loss.__get__(self.pipeline)
        self.pipeline.obtain_motion_representation = obtain_motion_representation.__get__(self.pipeline)
        
        for param in self.pipeline.unet.parameters():
            param.requires_grad = False
        for param in self.pipeline.controlnet.parameters():
            param.requires_grad = False
        
        self.pipeline.input_config,  self.pipeline.unet.input_config = config,  config
        self.pipeline.unet = prep_unet_attention(self.pipeline.unet,self.pipeline.input_config.motion_guidance_blocks)
        self.pipeline.unet = prep_unet_conv(self.pipeline.unet)
        self.pipeline.scheduler.customized_set_timesteps(config.inference_steps, config.guidance_steps,config.guidance_scale,device=device,timestep_spacing_type = "uneven")

        self.config  = config

    async def asyn_forward(self, websocket,message):
        import json
        response = await asyncio.get_running_loop().run_in_executor(self.executor, self.dsso_forward, message)
        await websocket.send(json.dumps(response))

    def dsso_init(self,req:Dict = None)->bool:
        pass
        
    def dsso_reload_conf(self,conf:ServerConfig):
        pass

    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {}
        example_infor = request
        self.config.video_path = example_infor["video_path"]
        self.config.condition_image_path_list = example_infor["condition_image_paths"]
        self.config.image_index = example_infor.get("image_index",[0])
        assert len(self.config.image_index) == len(self.config.condition_image_path_list)
        self.config.new_prompt = example_infor["new_prompt"] + self.config.get("positive_prompt", "")
        self.config.controlnet_scale = example_infor.get("controlnet_scale", 1.0)
        self.pipeline.input_config,  self.pipeline.unet.input_config = self.config,  self.config  # update config
        
        #  perform motion representation extraction
        seed_motion = example_infor.get("seed", self.args.default_seed) 
        generator = torch.Generator(device=self.pipeline.device)
        generator.manual_seed(seed_motion)
        if not os.path.exists(self.args.motion_representation_save_dir):
            os.makedirs(self.args.motion_representation_save_dir)
        motion_representation_path = os.path.join(self.args.motion_representation_save_dir,  os.path.splitext(os.path.basename(self.config.video_path))[0] + '.pt') 
        self.pipeline.obtain_motion_representation(generator= generator, motion_representation_path = motion_representation_path, use_controlnet=True,) 
        
        # perform video generation
        seed = seed_motion # can assign other seed here
        generator = torch.Generator(device=self.pipeline.device)
        generator.manual_seed(seed)
        self.pipeline.input_config.seed = seed
        videos = self.pipeline.sample_video(generator = generator, add_controlnet=True,)

        videos = rearrange(videos, "b c f h w -> b f h w c")
        save_path = os.path.join(self.args.generated_videos_save_dir, os.path.splitext(os.path.basename(self.config.video_path))[0]
                                    + "_" + self.config.new_prompt.strip().replace(' ', '_') + str(seed_motion) + "_" +str(seed)+'.mp4')                                        
        videos_uint8 = (videos[0] * 255).astype(np.uint8)
        imageio.mimwrite(save_path, videos_uint8, fps=8)



        output_map['output'] = self.uploader.upload_video(save_path)
        output_map['state'] = 'finished'
        return output_map

    



