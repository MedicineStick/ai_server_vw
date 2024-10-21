import os
import io
import logging
import numpy
import torch
import torchaudio
from pydub import AudioSegment
import ffmpeg
import ffmpeg
import io
import tempfile
import os
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt

def format_number(num):
    return f"/{num:05}.jpg"

import cv2
import requests
import numpy as np
from io import BytesIO

def load_image_cv2(image_path_or_url):
    # Check if it's a URL (starts with 'http')
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        if response.status_code == 200:
            image_data = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        else:
            raise Exception(f"Failed to load image from URL: {image_path_or_url}")
    else:  # Assume it's a local file
        image = cv2.imread(image_path_or_url)
        if image is None:
            raise Exception(f"Failed to load local image: {image_path_or_url}")
    
    # Convert from BGR to RGB (OpenCV loads in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def apply_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    """
    Apply a boolean mask to an image and output the image with the mask overlay.
    
    Args:
        image (np.ndarray): The input image (height x width x channels).
        mask (np.ndarray): The boolean mask with shape (1, h, w) or (h, w).
        color (tuple): The color to apply for the mask overlay (default is green).
        alpha (float): The transparency factor for the mask overlay (default is 0.5).
        
    Returns:
        np.ndarray: The image with the mask applied.
    """
    # Ensure mask is of shape (h, w) by squeezing the first dimension if needed
    mask = np.squeeze(mask)  # Shape becomes (h, w)

    # Convert the boolean mask to an integer mask (0s and 1s)
    mask = mask.astype(np.uint8)

    # Create a colored overlay using the mask
    colored_overlay = np.zeros_like(image, dtype=np.uint8)
    colored_overlay[mask == 1] = color

    # Combine the image with the colored overlay using alpha blending
    output_image = cv2.addWeighted(image, 1 - alpha, colored_overlay, alpha, 0)
    return output_image


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    
def generate_video_from_frames(image_paths, masks, output_video_path, fps=30):
    # Check if the number of images and masks are equal
    if len(image_paths) != len(masks):
        raise ValueError("The number of images and masks must be the same.")
        
    
    # Read the first frame to get video dimensions
    first_frame = cv2.imread(image_paths[0])
    height, width, _ = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for image_path, mask in zip(image_paths, masks):
        # Read the image frame
        frame = cv2.imread(image_path)
        
        # Apply the mask to the frame (assuming mask is binary)
        mask = np.squeeze(mask)
        mask_uint8 = (mask.astype(np.uint8)) * 255
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask_uint8)
        
        # Write the masked frame to the video
        video_writer.write(masked_frame)
    
    # Release the video writer
    video_writer.release()

def download_image(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as file:
            file.write(response.content)
        print(f"Image successfully downloaded: {local_path}")
    else:
        print(f"Failed to retrieve image. HTTP Status code: {response.status_code}")

def bytes_from_video_file(video_file_path):
    # Create a temporary file to store the output
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file_path = temp_file.name

    try:
        # Use ffmpeg to process the video and save it to the temporary file
        ffmpeg.input(video_file_path).output(temp_file_path, vcodec='libx264', acodec='aac').run(overwrite_output=True)
        
        # Read the temporary file into a BytesIO buffer
        with open(temp_file_path, 'rb') as f:
            video_bytes = f.read()
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return video_bytes
    except Exception as e:
        # Clean up the temporary file in case of an error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise e


def get_speech_timestamps_silero_vad(
    audio_file:str,
    sampling_rate_:int,
    vad_dir_:str,
    )->list:
        logging.info("Loading VAD model...")
        model, utils = torch.hub.load(repo_or_dir=vad_dir_,
                                    model='silero_vad',
                                    source='local',
                                    force_reload=False,
                                    onnx=True)

        (get_speech_timestamps,
        _,
        read_audio,
        _,
        _) = utils
        wav = read_audio(audio_file, sampling_rate=sampling_rate_)
        #/home/tione/notebook/lskong2/projects/2.tingjian/silero-vad-master/utils_vad.py 
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate_)
        return speech_timestamps

def audio_preprocess(
    audio_file:str,
    output_audio:str,
    ffmpeg_:str,
    sampling_rate_:int
):
    logging.info("Loading audio...")
    cmds = [ffmpeg_,'-y','-i',audio_file,'-ac 1','-ar',str(sampling_rate_),output_audio]
    cmd = ' '.join(cmds)
    logging.info(cmd)
    logging.info("Resampling...")
    os.system(cmd)

def process_timestamps(timestamps_:list)->list:
    output_list = []
    for time_ in timestamps_:
        start_ = time_['start']
        end_ = time_['end']
        output_list.append([start_,end_])
    return output_list

def trim_audio(
    audio_:str,
    vad_list_:list[list[float]],
    min_combine_sents_sec_sample:int
    )->list[torch.tensor]:

    waveform,_ = torchaudio.load(audio_,normalize=True)
    current_sample = 0
    cut_waveform = torch.zeros([1, 0], dtype=torch.int32)
    output_tensors = []
    for i in range(0,len(vad_list_)):
        s_e = vad_list_[i]
        start_sample = s_e[0]
        end_sample = s_e[1]
        current_sample+=(end_sample-start_sample)

        if current_sample>min_combine_sents_sec_sample:
            cut_waveform = torch.concat((cut_waveform,waveform[:,start_sample:end_sample]),dim=1)
            output_tensors.append(cut_waveform)
            cut_waveform = torch.zeros([1, 0], dtype=torch.int32)
            current_sample = 0
        else:
            cut_waveform = torch.concat((cut_waveform,waveform[:,start_sample:end_sample]),dim=1)

            if i == len(vad_list_)-1:
                output_tensors.append(cut_waveform)
            else:
                pass

    return output_tensors

def bytes_from_audio_file(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()
    
def bytes_from_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()

def bytes_from_audio(audio_array):
    buffer = io.BytesIO()
    numpy.save(buffer, audio_array)
    buffer.seek(0)
    return buffer.read()

def bytes_from_file(file_path):
    with open(file_path, 'rb') as file:
        return file.read()


def bytes_from_audio_tensor(audio_tensor:torch.Tensor, sample_rate=44100, format='wav'):
    """
    Convert an audio tensor to bytes.

    Args:
        audio_tensor (torch.Tensor): A 1D audio tensor.
        sample_rate (int): The sample rate of the audio.
        format (str): The audio format ('wav' or 'mp3').

    Returns:
        bytes: The audio data as bytes.
    """
    from scipy.io.wavfile import write

    if format not in ['wav', 'mp3']:
        raise ValueError("Format must be 'wav' or 'mp3'")

    # Convert tensor to numpy array
    audio_array = audio_tensor.numpy()

    # Save audio array to buffer
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_array)

    # Get byte data
    audio_bytes = buffer.getvalue()

    return audio_bytes

# Example usage:
# Assuming you have a 1D audio tensor `audio_tensor`
# audio_tensor = torch.randn(44100)  # 1 second of audio at 44100 Hz
# audio_bytes = bytes_from_audio_tensor(audio_tensor)


class CosUploader:
    def __init__(self,mode:int):
        secret_id = None
        secret_key = None
        self.bucket = None
        if mode ==0:
            secret_id = ""
            secret_key = ""
            self.bucket = 'inno-project1-1316407986'
        else:
            secret_id = ''
            secret_key = ''
            self.bucket='dsso-di-icp-prod-1322412301'

        region = 'ap-shanghai'
        token = None
        scheme = 'https'
        self.prefix = '/lskong2/service_upload'
        from qcloud_cos import CosConfig
        from qcloud_cos import CosS3Client
        config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
        self.client = CosS3Client(config)

    def upload_image(self, image, key: str = None) -> str:
        if key is None:
            import uuid
            key = str(uuid.uuid4()) + '.png'
        if len(self.prefix) > 0:
            key = os.path.join(self.prefix, key)
        from qcloud_cos.cos_exception import CosClientError, CosServiceError

        # 使用高级接口断点续传，失败重试时不会上传已成功的分块(这里重试10次)
        for i in range(0, 10):
            try:
                response = self.client.put_object(
                    Bucket=self.bucket,
                    Body=bytes_from_image(image),
                    Key=key)
                url = self.client.get_object_url(
                    Bucket=self.bucket,
                    Key=key,
                )
                return url
            except (CosClientError, CosServiceError) as e:
                logging.error(f"failed to upload image: {e}")
        return ""
    
    def upload_audio(self, audio, sample_rate:int=16000 ,key: str = None) -> str:
        if key is None:
            import uuid
            key = str(uuid.uuid4()) + '.wav'
        if len(self.prefix) > 0:
            key = os.path.join(self.prefix, key)
        from qcloud_cos.cos_exception import CosClientError, CosServiceError

        # 使用高级接口断点续传，失败重试时不会上传已成功的分块(这里重试10次)
        for i in range(0, 10):
            try:
                response = self.client.put_object(
                    Bucket=self.bucket,
                    Body=bytes_from_audio_file(audio),
                    Key=key)
                url = self.client.get_object_url(
                    Bucket=self.bucket,
                    Key=key,
                )
                return url
            except (CosClientError, CosServiceError) as e:
                logging.error(f"failed to upload image: {e}")
        return ""

    def upload_video(self, video ,key: str = None) -> str:
        if key is None:
            import uuid
            key = str(uuid.uuid4()) + '.mp4'
        if len(self.prefix) > 0:
            key = os.path.join(self.prefix, key)
        from qcloud_cos.cos_exception import CosClientError, CosServiceError

        # 使用高级接口断点续传，失败重试时不会上传已成功的分块(这里重试10次)
        for i in range(0, 10):
            try:
                response = self.client.put_object(
                    Bucket=self.bucket,
                    Body=bytes_from_video_file(video),
                    Key=key)
                url = self.client.get_object_url(
                    Bucket=self.bucket,
                    Key=key,
                )
                return url
            except (CosClientError, CosServiceError) as e:
                logging.error(f"failed to upload image: {e}")
        return ""
    
    def upload_file(self, file_temp:str ,key: str = None) -> str:

        extension = file_temp[file_temp.rfind('.'):]
        if key is None:
            import uuid
            key = str(uuid.uuid4()) + extension
        if len(self.prefix) > 0:
            key = os.path.join(self.prefix, key)
        from qcloud_cos.cos_exception import CosClientError, CosServiceError

        # 使用高级接口断点续传，失败重试时不会上传已成功的分块(这里重试10次)
        for i in range(0, 10):
            try:
                response = self.client.put_object(
                    Bucket=self.bucket,
                    Body=bytes_from_file(file_temp),
                    Key=key)
                url = self.client.get_object_url(
                    Bucket=self.bucket,
                    Key=key,
                )
                return url
            except (CosClientError, CosServiceError) as e:
                logging.error(f"failed to upload image: {e}")
        return ""

