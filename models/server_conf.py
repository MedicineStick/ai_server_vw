
from omegaconf import OmegaConf


class ServerConfig:
    def __init__(self,config_file:str):
        self._load_config(config_file)

    def _load_config(self, config_file):
        config = OmegaConf.load(config_file)
        self.gpu_id = config["gpu_id"]

        self.ai_classification_path = config["ai_classification_path"]
        self.ai_classification_num_class = config["ai_classification_num_class"]
        self.ai_classification_problem_type = config["ai_classification_problem_type"]
        self.ai_classification_input_sz = config["ai_classification_input_sz"]
        self.ai_classification_mem = config["ai_classification_mem"]

        self.forgery_detection_b_image = config["forgery_detection_b_image"]
        self.forgery_detection_mem = config["forgery_detection_mem"]
        self.forgery_detection_pitch_size = config["forgery_detection_pitch_size"]
        self.forgery_detection_threshold = config["forgery_detection_threshold"]
        self.forgery_detection_path = config["forgery_detection_path"]

        self.warning_light_detection_mem = config["warning_light_detection_mem"]
        self.warning_light_detection_input_size = config["warning_light_detection_input_size"]
        self.warning_light_detection_threshold = config["warning_light_detection_threshold"]
        self.warning_light_detection_path = config["warning_light_detection_path"]
        self.warning_light_detection_class_num = config["warning_light_detection_class_num"]


        self.super_resolution_mem = config["super_resolution_mem"]
        self.super_resolution_tile_pad = config["super_resolution_tile_pad"]
        self.super_resolution_pre_pad = config["super_resolution_pre_pad"]
        self.super_resolution_outscale = config["super_resolution_outscale"]
        self.super_resolution_tile = config["super_resolution_tile"]
        self.super_resolution_netscale = config["super_resolution_netscale"]
        self.super_resolution_path = config["super_resolution_path"]
        self.super_resolution_mode = config["super_resolution_mode"]


        self.ai_meeting_mem = config["ai_meeting_mem"]
        self.ai_meeting_temp_path = config["ai_meeting_temp_path"]
        self.ai_meeting_output_encoding = config["ai_meeting_output_encoding"]
        self.ai_meeting_ffmpeg_file = config["ai_meeting_ffmpeg_file"]
        self.ai_meeting_vad_dir = config["ai_meeting_vad_dir"]
        self.ai_meeting_whisper_model_name = config["ai_meeting_whisper_model_name"]
        self.ai_meeting_asr_model_path = config["ai_meeting_asr_model_path"]
        self.ai_meeting_language = config["ai_meeting_language"]
        self.ai_meeting_min_combine_sents_sec = config["ai_meeting_min_combine_sents_sec"]
        self.ai_meeting_max_decode = config["ai_meeting_max_decode"]
        self.ai_meeting_if_write_asr = config["ai_meeting_if_write_asr"]
        self.ai_meeting_transcribe_gap = config["ai_meeting_transcribe_gap"]
        self.ai_meeting_beam_size = config["ai_meeting_beam_size"]
        self.ai_meeting_if_greedy = config["ai_meeting_if_greedy"]
        self.ai_meeting_if_msdp = config["ai_meeting_if_msdp"]
        self.ai_meeting_n_speakers = config["ai_meeting_n_speakers"]
        self.ai_meeting_manifest_file = config["ai_meeting_manifest_file"]
        self.ai_meeting_if_write_msdp = config["ai_meeting_if_write_msdp"]
        self.ai_meeting_if_summary = config["ai_meeting_if_summary"]
        self.ai_meeting_llm_path = config["ai_meeting_llm_path"]
        self.ai_meeting_vad_context_sec = config["ai_meeting_vad_context_sec"]
        self.ai_meeting_prompt_en_summarize = config["ai_meeting_prompt_en_summarize"]
        self.ai_meeting_prompt_cn_summarize = config["ai_meeting_prompt_cn_summarize"]
        self.ai_meeting_prompt_zh_punctuation = config["ai_meeting_prompt_zh_punctuation"]
        self.ai_meeting_max_tokens = config["ai_meeting_max_tokens"]
        self.ai_meeting_if_write_summary = config["ai_meeting_if_write_summary"]
        self.ai_meeting_if_write_dia_summary = config["ai_meeting_if_write_dia_summary"]
        self.ai_meeting_diar_infer_meeting_cfg = config["ai_meeting_diar_infer_meeting_cfg"]
        self.ai_meeting_file_cluster = config["ai_meeting_file_cluster"]
        self.ai_meeting_output_dia = config["ai_meeting_output_dia"]
        self.ai_meeting_supported_sampling_rate = config["ai_meeting_supported_sampling_rate"]
        self.ai_meeting_chatbot_url = config["ai_meeting_chatbot_url"]

        self.online_asr_model_en = config["online_asr_model_en"]
        self.online_asr_model_cn = config["online_asr_model_cn"]
        self.online_asr_model_de = config["online_asr_model_de"]
        self.online_asr_sample_rate = config["online_asr_sample_rate"]
        self.online_asr_mem = config["online_asr_mem"]


        self.tts_en_mem = config["tts_en_mem"]
        self.tts_cn_mem = config["tts_cn_mem"]
        self.translation_mem = config["translation_mem"]
        self.translation_model_path = config["translation_model_path"]


        self.video_generation_mem = config["video_generation_mem"]
        self.video_generation_t5_device = config["video_generation_t5_device"]
        self.video_generation_vae_device = config["video_generation_vae_device"]
        self.video_generation_sd_device = config["video_generation_sd_device"]

        #realtime_asr_whisper_model_name

        self.realtime_asr_whisper_model_name = config["realtime_asr_whisper_model_name"]
        self.realtime_asr_min_combine_sents_sec = config["realtime_asr_min_combine_sents_sec"]
        self.realtime_asr_model_sample = config["realtime_asr_model_sample"]
        self.realtime_asr_gap_ms = config["realtime_asr_gap_ms"]
        self.realtime_asr_beam_size = config["realtime_asr_beam_size"]
        self.realtime_asr_min_silence_duration_ms = config["realtime_asr_min_silence_duration_ms"]

        self.sam2_device_id = config["sam2_device_id"]
        self.sam2_video_dir = config["sam2_video_dir"]
        self.sam2_checkpoint = config["sam2_checkpoint"]
        self.sam2_model_cfg = config["sam2_model_cfg"]
        self.sam2_vis_frame_stride = config["sam2_vis_frame_stride"]


        self.sam1_device_id = config["sam1_device_id"]
        self.sam1_checkpoint = config["sam1_checkpoint"]
        self.sam1_model_type = config["sam1_model_type"]


        self.cos_uploader_mode = config["cos_uploader_mode"]
        self.time_blocker = config["time_blocker"]

        self.realtime_asr_max_length_ms_chatbot = config["realtime_asr_max_length_ms_chatbot"]
        self.realtime_asr_min_silence_duration_ms_chatbot = config["realtime_asr_min_silence_duration_ms_chatbot"]
        self.realtime_asr_adaptive_thresholding_chatbot = config["realtime_asr_adaptive_thresholding_chatbot"]


        self.ai_meeting_llm_timeout  = config["ai_meeting_llm_timeout"]
        self.ai_meeting_retry_count = config["ai_meeting_retry_count"]
        self.realtime_asr_llm_timeout = config["realtime_asr_llm_timeout"]
        self.realtime_asr_retry_count = config["realtime_asr_retry_count"]

        self.motion_clone_device = config["motion_clone_device"]

        self.jumpcutter_temp_floder = config["jumpcutter_temp_floder"]
        self.jumpcutter_sounded_speed = config["jumpcutter_sounded_speed"]
        self.jumpcutter_silent_speed = config["jumpcutter_silent_speed"]
        self.jumpcutter_output = config["jumpcutter_output"]


        
        


