import torch
from myapp.dsso_server import DSSO_SERVER
from myapp.server_conf import ServerConfig
from myapp.dsso_util import download_image,generate_video_from_frames,CosUploader
from typing import Dict
from sam2.build_sam import build_sam2_video_predictor
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import logging  # OpenCV for image processing
import shutil



class Sam2(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        print("--->initialize Sam2...")
        super().__init__()
        self.conf = conf
        self._need_mem = self.conf.ai_classification_mem
        self.device = torch.device(self.conf.sam2_device_id)
        self.predictor = build_sam2_video_predictor(self.conf.sam2_model_cfg, self.conf.sam2_checkpoint, device=self.device)
        self.temp_video_path = self.conf.sam2_video_dir
        self.sam2_vis_frame_stride = self.conf.sam2_vis_frame_stride
        self.inference_state = None
        try:
            self.uploader = CosUploader(self.conf.super_resolution_mode)
        except Exception as e:
            self.uploader = None
            logging.info(f'no cos uploader found: {e}')

    def dsso_init(self,req:Dict = None)->bool:
        pass
        
    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self.device = torch.device(self.conf.gpu_id)


    def dsso_forward(self, request: Dict) -> Dict:
        output_map = {}
        name = request["task_name"].replace(' ','').strip()
        temp_video = self.temp_video_path + '/'+name+'.mp4'
        temp_masked_video = self.temp_video_path + '/'+name+'_masked.mp4'
        temp_images_path = self.temp_video_path+'/'+name
        if not os.path.exists(temp_images_path):
            os.mkdir(temp_images_path)

        video = request["video"].strip()
        if 'http' in video:
            download_image(video,temp_video)
        else:
            shutil.copy(video,temp_video)

        labels_points = request["labels_points"]
        labels = request["labels"]

        ann_frame_idx = request["ann_frame_idx"]  # the frame index we interact with
        ann_obj_id = request["ann_obj_id"]  # give a unique id to each object we interact with (it can be any integers)

        # Let's add a positive click at (x, y) = (210, 350) to get started
        points = np.array(labels_points, dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array(labels, np.int32)

        ffmpeg_cmd = "ffmpeg -i "+temp_video+ " -q:v 2 -start_number 0 "+temp_images_path+"/\'%05d.jpg\'"
        print(ffmpeg_cmd)
        os.system(ffmpeg_cmd)

        frame_names = [
            p for p in os.listdir(temp_images_path)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        if self.inference_state ==None:
            pass
        else:
            self.predictor.reset_state(self.inference_state)

        self.inference_state = self.predictor.init_state(video_path=temp_images_path)

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        masks = []
        for out_frame_idx in tqdm(range(0, len(frame_names))):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            #plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for _, out_mask in video_segments[out_frame_idx].items():
                #show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                masks.append(out_mask)
        
        frame_names = [ temp_images_path+'/'+image_name for image_name in frame_names]
        generate_video_from_frames(frame_names,masks,temp_masked_video)


        output_map['video'] = self.uploader.upload_video(temp_masked_video)
        output_map['state'] = 'finished'
        return output_map,True
