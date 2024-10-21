import os
import time
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm
import sys
sys.path.append("./third_party/")

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype
from myapp.dsso_server import DSSO_SERVER
from models.server_conf import ServerConfig
from typing import Dict
from models.dsso_util import CosUploader,download_image
import logging

class Video_Generation(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        print("--->initialize Video_Generation...")
        super().__init__()
        self.conf = conf
        self._need_mem = self.conf.video_generation_mem
        self.t5_device = self.conf.video_generation_t5_device
        self.vae_device = self.conf.video_generation_vae_device
        self.sd_device = self.conf.video_generation_sd_device
        try:
            self.uploader = CosUploader(self.conf.super_resolution_mode)
        except Exception as e:
            self.uploader = None
            logging.info(f'no cos uploader found: {e}')
        pass
        self.cfg = {'resolution': '480p', 'aspect_ratio': '9:16', 'num_frames': '4s', 'fps': 24, 'frame_interval': 1, 'save_fps': 24, 'save_dir': './samples/samples/', 'seed': 42, 'batch_size': 1, 'multi_resolution': 'STDiT2', 'dtype': 'bf16', 'condition_frame_length': 5, 'align': 5, 'model': {'type': 'STDiT3-XL/2', 'from_pretrained': 'hpcai-tech/OpenSora-STDiT-v3', 'qk_norm': True, 'enable_flash_attn': False, 'enable_layernorm_kernel': False}, 'vae': {'type': 'OpenSoraVAE_V1_2', 'from_pretrained': 'hpcai-tech/OpenSora-VAE-v1.2', 'micro_frame_size': 17, 'micro_batch_size': 4}, 'text_encoder': {'type': 't5', 'from_pretrained': 'DeepFloyd/t5-v1_1-xxl', 'model_max_length': 300}, 'scheduler': {'type': 'rflow', 'use_timestep_transform': True, 'num_sampling_steps': 30, 'cfg_scale': 7.0}, 'aes': 6.5, 'flow': None, 'config': 'configs/opensora-v1-2/inference/sample.py', 'flash_attn': False, 'layernorm_kernel': False, 'prompt_as_path': False, 'prompt': []}
        # 'A car is driving in the forest.'

        cfg_dtype = self.cfg.get("dtype", "fp32")
        assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
        self.dtype = to_torch_dtype(self.cfg.get("dtype", "bf16"))

        
    def dsso_init(self,req:Dict = None)->bool:
        pass
        
    
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = self.conf.video_generation_mem


    def dsso_forward(self, request: Dict) -> Dict:
        
        request["image_start"] = request["image_start"].strip()
        request["image_end"] = request["image_end"].strip()
        request["continue_url"] = request["continue_url"].strip()
        
        if len(request["image_start"])==0:
            if  len(request["continue_url"])==0:
                self.cfg['prompt'].append(request['prompt'])
            else:
                postfix = "{\"reference_path\": \""+request["continue_url"]+"\",\"mask_strategy\": \"0,0,0,-8,8\"}"
                self.cfg['prompt'].append(request['prompt']+postfix)
        else:
            if len(request["image_end"])==0:
                if "http" in request["image_start"]:
                    download_image(request["image_start"],"samples/samples/pictures/1.jpg")
                    postfix = "{\"reference_path\": \"samples/samples/pictures/1.jpg\",\"mask_strategy\": \"0\"}"
                    self.cfg['prompt'].append(request['prompt']+postfix)
                else:
                    postfix = "{\"reference_path\": \""+request["image_start"]+"\",\"mask_strategy\": \"0\"}"
                    self.cfg['prompt'].append(request['prompt']+postfix)
            else:
                if "http" in request["image_start"]:
                    download_image(request["image_start"],"samples/samples/pictures/1.jpg")
                    download_image(request["image_end"],"samples/samples/pictures/2.jpg")
                    postfix = "{\"reference_path\": \"samples/samples/pictures/1.jpg;samples/samples/pictures/2.jpg\",\"mask_strategy\": \"0;0,1,0,-1,1\"}"
                    self.cfg['prompt'].append(request['prompt']+postfix)
                else:
                    postfix = "{\"reference_path\": \""+request["image_start"]+";"+request["image_end"]+"\",\"mask_strategy\": \"0;0,1,0,-1,1\"}"
                    self.cfg['prompt'].append(request['prompt']+postfix)
        self.cfg['num_frames'] = str(request['num_frames'])+'s'
        self.cfg['aspect_ratio'] = request['ratio'].strip()
        
        vae_model = build_module(self.cfg['vae'], MODELS).to(self.vae_device, self.dtype).eval()
        
        text_encoder_model = build_module(self.cfg['text_encoder'], MODELS, device=self.t5_device)

        model = None

        output_map = {}

        torch.set_grad_enabled(False)
        # ======================================================
        # configs & runtime variables
        # ======================================================
        # == parse configs ==
        #cfg = parse_configs(training=False)

        # == device and dtype ==
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # == init distributed env ==
        if is_distributed():
            colossalai.launch_from_torch({})
            coordinator = DistCoordinator()
            enable_sequence_parallelism = coordinator.world_size > 1
            if enable_sequence_parallelism:
                set_sequence_parallel_group(dist.group.WORLD)
        else:
            coordinator = None
            enable_sequence_parallelism = False
        set_random_seed(seed =self.cfg.get("seed", 1024))

        # == init logger ==
        logger = create_logger()
        logger.info("Inference configuration:\n %s", pformat(self.cfg))
        verbose = self.cfg.get("verbose", 1)
        progress_wrap = tqdm if verbose == 1 else (lambda x: x)

        # ======================================================
        # build model & load weights
        # ======================================================
        #logger.info("Building models...")
        # == build text-encoder and vae ==
        
        

        # == prepare video size ==
        image_size = self.cfg.get("image_size", None)
        if image_size is None:
            resolution = self.cfg.get("resolution", None)
            aspect_ratio = self.cfg.get("aspect_ratio", None)
            assert (
                resolution is not None and aspect_ratio is not None
            ), "resolution and aspect_ratio must be provided if image_size is not provided"
            image_size = get_image_size(resolution, aspect_ratio)
        num_frames = get_num_frames(self.cfg['num_frames'])
        # == build diffusion model ==
        input_size = (num_frames, *image_size)
        latent_size = vae_model.get_latent_size(input_size)
        model = (
            build_module(
                self.cfg['model'],
                MODELS,
                input_size=latent_size,
                in_channels=vae_model.out_channels,
                caption_channels=text_encoder_model.output_dim,
                model_max_length=text_encoder_model.model_max_length,
                enable_sequence_parallelism=enable_sequence_parallelism,
            )
            .to(self.sd_device, self.dtype)
            .eval()
        )
        text_encoder_model.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

        # == build scheduler ==
        scheduler = build_module(self.cfg['scheduler'], SCHEDULERS)

        # ======================================================
        # inference
        # ======================================================
        # == load prompts ==
        prompts = self.cfg.get("prompt", None)
        start_idx = self.cfg.get("start_index", 0)
        if prompts is None:
            if self.cfg.get("prompt_path", None) is not None:
                prompts = load_prompts(self.cfg['prompt_path'], start_idx, self.cfg.get("end_index", None))
            else:
                prompts = [self.cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

        # == prepare reference ==
        reference_path = self.cfg.get("reference_path", [""] * len(prompts))
        mask_strategy = self.cfg.get("mask_strategy", [""] * len(prompts))
        assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
        assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

        # == prepare arguments ==
        fps = self.cfg['fps']
        save_fps = self.cfg.get("save_fps", fps // self.cfg.get("frame_interval", 1))
        multi_resolution = self.cfg.get("multi_resolution", None)
        batch_size = self.cfg.get("batch_size", 1)
        num_sample = self.cfg.get("num_sample", 1)
        loop = self.cfg.get("loop", 1)
        condition_frame_length = self.cfg.get("condition_frame_length", 5)
        condition_frame_edit = self.cfg.get("condition_frame_edit", 0.0)
        align = self.cfg.get("align", None)

        save_dir = self.cfg['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        sample_name = self.cfg.get("sample_name", None)
        prompt_as_path = self.cfg.get("prompt_as_path", False)

        # == Iter over all samples ==
        for i in progress_wrap(range(0, len(prompts), batch_size)):
            # == prepare batch prompts ==
            batch_prompts = prompts[i : i + batch_size]
            ms = mask_strategy[i : i + batch_size]
            refs = reference_path[i : i + batch_size]

            # == get json from prompts ==
            batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
            original_batch_prompts = batch_prompts

            # == get reference for condition ==
            refs = collect_references_batch(refs, vae_model, image_size)

            # == multi-resolution info ==
            model_args = prepare_multi_resolution_info(
                multi_resolution, len(batch_prompts), image_size, num_frames, fps, self.sd_device, self.dtype
            )

            # == Iter over number of sampling for one prompt ==
            for k in range(num_sample):
                # == prepare save paths ==
                save_paths = ['./samples/samples/sample_0000']

                # NOTE: Skip if the sample already exists
                # This is useful for resuming sampling VBench
                if prompt_as_path and all_exists(save_paths):
                    continue

                # == process prompts step by step ==
                # 0. split prompt
                # each element in the list is [prompt_segment_list, loop_idx_list]
                batched_prompt_segment_list = []
                batched_loop_idx_list = []
                for prompt in batch_prompts:
                    prompt_segment_list, loop_idx_list = split_prompt(prompt)
                    batched_prompt_segment_list.append(prompt_segment_list)
                    batched_loop_idx_list.append(loop_idx_list)

                # 1. refine prompt by openai
                if self.cfg.get("llm_refine", False):
                    # only call openai API when
                    # 1. seq parallel is not enabled
                    # 2. seq parallel is enabled and the process is rank 0
                    if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                            batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                    # sync the prompt if using seq parallel
                    if enable_sequence_parallelism:
                        coordinator.block_all()
                        prompt_segment_length = [
                            len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                        ]

                        # flatten the prompt segment list
                        batched_prompt_segment_list = [
                            prompt_segment
                            for prompt_segment_list in batched_prompt_segment_list
                            for prompt_segment in prompt_segment_list
                        ]

                        # create a list of size equal to world size
                        broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                        dist.broadcast_object_list(broadcast_obj_list, 0)

                        # recover the prompt list
                        batched_prompt_segment_list = []
                        segment_start_idx = 0
                        all_prompts = broadcast_obj_list[0]
                        for num_segment in prompt_segment_length:
                            batched_prompt_segment_list.append(
                                all_prompts[segment_start_idx : segment_start_idx + num_segment]
                            )
                            segment_start_idx += num_segment

                # 2. append score
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = append_score_to_prompts(
                        prompt_segment_list,
                        aes =self.cfg.get("aes", None),
                        flow =self.cfg.get("flow", None),
                        camera_motion =self.cfg.get("camera_motion", None),
                    )

                # 3. clean prompt with T5
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

                # 4. merge to obtain the final prompt
                batch_prompts = []
                for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                    batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

                # == Iter over loop generation ==
                video_clips = []
                for loop_i in range(loop):
                    # == get prompt for loop i ==
                    batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                    # == add condition frames for loop ==
                    if loop_i > 0:
                        refs, ms = append_generated(
                            vae_model, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                        )

                    # == sampling ==
                    z = torch.randn(len(batch_prompts), vae_model.out_channels, *latent_size, device=self.t5_device, dtype=self.dtype)
                    # z (batch,c_out,t,h,w)
                    masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                    samples = scheduler.sample(
                        model,
                        text_encoder_model,
                        z=z,
                        prompts=batch_prompts_loop,
                        device=self.t5_device,
                        additional_args=model_args,
                        progress=verbose >= 2,
                        mask=masks,
                    )
                    samples = vae_model.decode(samples.to(self.dtype), num_frames=num_frames)
                    video_clips.append(samples)

                # == save samples ==
                if is_main_process():
                    for idx, batch_prompt in enumerate(batch_prompts):
                        if verbose >= 2:
                            logger.info("Prompt: %s", batch_prompt)
                        save_path = save_paths[idx]
                        video = [video_clips[i][idx] for i in range(loop)]
                        for i in range(1, loop):
                            video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
                        video = torch.cat(video, dim=1)
                        save_path = save_sample(
                            video,
                            fps=save_fps,
                            save_path=save_path,
                            verbose=verbose >= 2,
                        )
                        if save_path.endswith(".mp4") and self.cfg.get("watermark", False):
                            time.sleep(1)  # prevent loading previous generated video
                            add_watermark(save_path)
            start_idx += len(batch_prompts)
        logger.info("Inference finished.")

        if request["if_sr"]:
            output_map['video'] = "./samples/samples/sample_0000.mp4"    
        else:
            output_map['video'] = self.uploader.upload_video("./samples/samples/sample_0000.mp4")
        return output_map,True
        
