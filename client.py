# Copyright 2023 MOSEC Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import websockets
import wave
import base64
import sys
import asyncio
import scipy
import re
import numpy as np 
from scipy.io.wavfile import write
WS_URL = 'ws://172.27.11.12:9501/ws'  # Replace with your WebSocket URL
import pyaudio
import torchaudio

def inner_func(audio:int,name:str):
    print(audio)
    print(name)

def func_args(**kwargs)->dict:
    inner_func(**kwargs)

def test_args():
    func_args(audio = 1,name = "Kay")

def vits_conversion():
    from models.dsso_util import audio_preprocess,get_speech_timestamps_silero_vad,process_timestamps,trim_audio
    import torchaudio
    audio_url = "/home/tione/notebook/lskong2/projects/2.tingjian/test_set/luoxiang.mp3"
    output_path = "temp/audio/"
    resample_wav = output_path+'/temp.wav'
    audio_preprocess(audio_url,
                     resample_wav,
                     "/home/tione/notebook/lskong2/softwares/ffmpeg-6.1/ffmpeg",
                     16000
                     )
    speech_timestamps = get_speech_timestamps_silero_vad(
                    audio_file=resample_wav,
                    sampling_rate_=16000,
                    vad_dir_="../2.tingjian/silero-vad-master/"
                    )
    speech_timestamps_list = process_timestamps(speech_timestamps)
    output_tensors = trim_audio(
                audio_=resample_wav,
                vad_list_=speech_timestamps_list,
                min_combine_sents_sec_sample=15
                )
    
    for i in range(0,len(output_tensors)):
        torchaudio.save("temp/audio/dataset_raw/"+str(i)+".wav", output_tensors[i], 16000)







def test3():
    from models.dsso_util import CosUploader
    
    file_a = "../2.tingjian/test_set/temp_resample_concated.wav"
    #waveform,sr = torchaudio.load("../2.tingjian/test_set/temp_resample_concated.wav",normalize=True)
    #print(waveform.shape)
    #print(sr)
    uploader = CosUploader(0)
    url1 = uploader.upload_audio(file_a)
    print(url1)


def test2():
    from models.server_conf import ServerConfig
    from myapp.ai_meeting_chatbot import AI_Meeting_Chatbot
    data = {"project_name":"ai_meeting_assistant_chatbot",
            "task_id":"fia967c_2024_06_2517_21_39_165_d",
            "audio_url":"/home/tione/notebook/lskong2/projects/AI_Server_V3/temp/results/ai_meeting_results/fia967c_2024_06_2517_21_39_165/ori.wav",
            #"audio_url":"./temp/voice20240124.m4a",
            #"audio_url":"./temp/luoxiang.wav",
            "task_type":1,
            "task_state":0,
            "lang":"zh",
            "recognize_speakers":1,
            "speaker_num":3 
            }
    global_conf = ServerConfig("./myapp/conf.yaml")
    model = AI_Meeting_Chatbot(global_conf)
    model.dsso_init(data)
    response,_ = model.dsso_forward_http(data)
    print(response)


def test1():
    from myapp.vits_tts_en import vits_tts
    from models.server_conf import ServerConfig
    data = {"project_name":"vits_tts",
            "text":"The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves.",
            "gender":1,  #0 woman, 1 man
            }
    global_conf = ServerConfig("./myapp/conf.yaml")
    model = vits_tts(global_conf)
    model.dsso_init()
    response,_ = model.dsso_forward_http(data)
    binary_data = base64.b64decode(response["audio_data"].encode())
        
    audio_array = np.frombuffer(binary_data, dtype=np.float32)
    write("./temp/11.wav", 22050, audio_array)

def test():
    from myapp.ai_classification import AI_Classification
    from myapp.warning_light_detection import warning_light_detection
    from myapp.forgery_detection import forgery_detection
    from myapp.vits_tts_en import vits_tts

    data = {"project_name":"warning_light_detection","image_url":"../6.DeepLearningModelServer/temp/1712561012318.png"}
    global_conf = ServerConfig("./myapp/conf.yaml")
    model = warning_light_detection(global_conf)
    model.dsso_init()
    output = model.dsso_forward_http(data)
    print(data,output)

    data = {"project_name":"forgery_detection","image_url":"../6.DeepLearningModelServer/temp/1712561012318.png"}
    model = forgery_detection(global_conf)
    model.dsso_init()
    output = model.dsso_forward_http(data)
    print(data,output)

    photos_path2 = "../3.forgery_detection/data/VOC2024_2/test/images/9139.png"
    data = {"project_name":"ai_classification","image_url":photos_path2}
    model = AI_Classification(global_conf)
    model.dsso_init()
    output = model.dsso_forward_http(data)
    print(data,output)

    

async def warning_detection1():

    data = {"project_name":"warning_light_detection","image_url":"./temp/isolation_warning_light_fake_2895.jpg"}
    encoded_data = json.dumps(data) #.encode("utf-8")
    
    async with websockets.connect(WS_URL) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(f"Received from server: {response}")


async def forgery():

    data = {"project_name":"forgery_detection","image_url":"../3.forgery_detection/temp/0.png"}
    encoded_data = json.dumps(data) #.encode("utf-8")
    
    async with websockets.connect(WS_URL) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(f"Received from server: {response}")



async def super_resolution():


    data = {"project_name":"super_resolution","image_url":"./temp/1.png"}
    encoded_data = json.dumps(data) #.encode("utf-8")
    
    async with websockets.connect(WS_URL) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(f"Received from server: {response}")





async def ai_class():
    photos_path2 = "../../data/AI_Classify/sdv1_5/train/class1/000_sdv5_00172.png"
    data = {"project_name":"ai_classification","image_url":photos_path2}
    encoded_data = json.dumps(data) #.encode("utf-8")
    import shutil
    shutil.copy(photos_path2,'temp/1.png')
    async with websockets.connect(WS_URL) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(f"Received from server: {response}")

    #http1 = urllib3.PoolManager()
    #r = http1.request('POST',url,body=encoded_data)
    #response = r.data.decode("utf-8")
    #output = json.loads(response)
    print(response)

async def ai_meeting():
    data = {"project_name":"ai_meeting_assistant",
            "task_id":"20240513_a",
            "audio_url":"/home/tione/notebook/lskong2/projects/2.tingjian/test_set/voice20240124.m4a",
            "task_type":1,
            "task_state":0,
            "lang":"en",
            
            }
    url = 'ws://127.0.0.1:8501/ws'
    encoded_data = json.dumps(data) #.encode("utf-8")
    
    async with websockets.connect(url) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(f"Received from server: {response}")

async def ai_meeting_chatbot():
    data = {"project_name":"ai_meeting_assistant_chatbot",
            "task_id":"lskong2_1213townhall",
            "audio_url":"./temp/1213townhall_firsthalf.m4a",
            #"audio_url":"./temp/voice20240124.m4a",
            #"audio_url":"./temp/luoxiang.wav",
            "task_type":1,
            "task_state":0,
            "lang":"en",
            "recognize_speakers":0,
            "speaker_num":6,
            "trans":1 
            }
    #data = {'project_name': 'ai_meeting_assistant_chatbot', 'audio_url': 'temp/results/ai_meeting_results/6122881163484439284/ori.wav', 'task_type': 1, 'task_id': '68228891663110586451a', 'lang': 'en', 'recognize_speakers': 1, 'speaker_num': 2, 'task_state': 0}
    encoded_data = json.dumps(data) #.encode("utf-8")
    
    async with websockets.connect(WS_URL) as websocket:
        print("----")
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(f"Received from server: {response}")

def ai_meeting_chatbot_offline():
    data = {"project_name":"ai_meeting_assistant_chatbot",
            "task_id":"lskong2_tesla_autopilot_1107",
            "audio_url":"./temp/tesla_autopilot.mp3",
            #"audio_url":"./temp/voice20240124.m4a",
            #"audio_url":"./temp/luoxiang.wav",
            "task_type":1,
            "task_state":0,
            "lang":"en",
            "recognize_speakers":0,
            "speaker_num":3,
            "trans":1 
            }
    from myapp.ai_meeting_chatbot import AI_Meeting_Chatbot
    from models.server_conf import ServerConfig
    global_conf = ServerConfig("./myapp/conf.yaml")
    model = AI_Meeting_Chatbot(global_conf)
    model.dsso_init(data)
    response,_ = model.dsso_forward_http(data)
    print(response)

async def online_asr():
    
    AUDIO_FILE_PATH = '../2.tingjian/test_set/temp_resample.wav'
    async with websockets.connect(WS_URL) as websocket:
        """
        while True:
            encoded_data = json.dumps({
                            'project_name':'online_asr',
                            'language_code': 'en',  # 0 for English, 1 for Chinese
                            'audio_data': count
                        })
            count+=1
            await websocket.send(encoded_data)
            response = await websocket.recv()
            print(f"Received from server: {response}")
        """
        with open(AUDIO_FILE_PATH, 'rb') as wf:
            wf.read(44)
            CHUNK = 4000
            while True:
                data = wf.read(CHUNK)
                if len(data) == 0:
                    break
                while len(data) > 0:
                    # Encode the audio data to base64
                    encoded_audio = base64.b64encode(data).decode()
                    # Prepare the JSON message with language code
                    encoded_data = json.dumps({
                        'project_name':'online_asr',
                        'language_code': 'en',  # 0 for English, 1 for Chinese
                        'audio_data': encoded_audio
                    })
                    await websocket.send(encoded_data)
                    response = await websocket.recv()
                    print(f"Received from server: {response}")
        
    
async def online_asr_en():
    async with websockets.connect(WS_URL) as websocket:

        #wf = wave.open('/home/tione/notebook/lskong2/projects/2.tingjian/test_set/tesla_autopilot.wav', "rb")
        #temp/1cdc7498c6d2b8dde71772e73e75af43.webm
        wf = open('/home/tione/notebook/lskong2/projects/2.tingjian/test_set/tesla_autopilot.wav', "rb")

        sample_rate = 48000
        input = {"project_name":"online_asr",
                   "language_code":'en',
                   "audio_data":None,
                   "state":"start",
                   "sample_rate":sample_rate
                   }

        await websocket.send(json.dumps(input))
        buffer_size = int(sample_rate * 2 * 0.2) # 0.2 seconds of audio
        print("buffer_size: ",buffer_size)
        buffer = 0
        while True:
            data = wf.read(buffer_size)
            buffer +=buffer_size
            print(buffer)
            if len(data) == 0:
                break
            encoded_audio = base64.b64encode(data).decode()
            input['audio_data'] = encoded_audio
            input['state'] = 'continue'

            await websocket.send(json.dumps(input))
            #exit(0)
            print (await websocket.recv())


        input['state'] = 'finished'
        await websocket.send(json.dumps(input))
        print (await websocket.recv())


async def online_asr_en_webm():
    async with websockets.connect(WS_URL) as websocket:

        audio = open('temp/a.webm', 'rb')
        data = audio.read()
        

        input = {"project_name":"online_asr_webm",
                   "language_code":'en',
                   "audio_data":None,
                   "state":"start",
                   "sample_rate":48000
                   }

        await websocket.send(json.dumps(input))
        encoded_audio = base64.b64encode(data).decode()
        input['audio_data'] = encoded_audio
        input['state'] = 'continue'
        await websocket.send(json.dumps(input))
        print (await websocket.recv())

        input['audio_data'] = None
        input['state'] = 'finished'
        await websocket.send(json.dumps(input))
        print (await websocket.recv())


async def online_asr_cn():
    async with websockets.connect(WS_URL) as websocket:

        wf = wave.open('/home/tione/notebook/lskong2/projects/2.tingjian/test_set/luoxiang_ac1.wav', "rb")
        sample_rate_ = 48000
        input = {"project_name":"online_asr",
                   "language_code":'cn',
                   "audio_data":None,
                   "state":"start",
                   'sample_rate':sample_rate_
                   }

        await websocket.send(json.dumps(input))
        buffer_size = int(sample_rate_ * 0.2) # 0.2 seconds of audio
        print("buffer_size: ",buffer_size)
        while True:
            data = wf.readframes(buffer_size)

            if len(data) == 0:
                break
            encoded_audio = base64.b64encode(data).decode()
            input['audio_data'] = encoded_audio
            input['state'] = 'continue'

            await websocket.send(json.dumps(input))
            #print (await websocket.recv())
            result = await websocket.recv()
            
            if len(result)>0 and "text" in result:
                resultd = json.loads(result)
                print(resultd)


        input['state'] = 'finished'
        await websocket.send(json.dumps(input))
        print (await websocket.recv())


async def vits_tts_en():
    data = {"project_name":"vits_tts_en",
            "text":"The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves.",
            "gender":1,  #0 woman, 1 man
            }
    output_audio = "./temp/vits_tts_en.wav"
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        binary_data = base64.b64decode(json.loads(response)["audio_data"].encode())
        
        audio_array = np.frombuffer(binary_data, dtype=np.float32)
        write(output_audio, 22050, audio_array)

async def vits_tts_cn():
    sentence = "遥望星空作文独自坐在乡间的小丘上"

    data = {"project_name":"vits_tts_cn",
            "text":sentence,
            "gender":1,  #0 woman, 1 man
            }
    output_audio = "./temp/vits_tts_cn.wav"
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=99999999999999) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()

        import scipy
        # 音频 -> base64
        #audio #Numpy 数组
        #binary_stream = audio.tobytes()
        #encoded_audio = base64.b64encode(binary_stream).decode()

        # base64 -> 音频
        binary_data = base64.b64decode(json.loads(response)["audio_data"].encode())
        audio_array = np.frombuffer(binary_data, dtype=np.float32)
        scipy.io.wavfile.write(output_audio, 16000, audio_array)
    
async def translation_zh2en():
    article_hi = "With the coming of Category and Product launch, the effectiveness of creative assets and category preference are crucial to our DSSO marketing promotion."
    data = {"project_name":"translation",
            "text":article_hi,
            "task":'zh2en'
            }
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(json.loads(response))

async def translation_en2zh():
    article_ar = "Sitting alone on a hill in the countryside, watching the sun gradually dim, listening to the birdsong gradually weakening, feeling the breeze gradually getting cooler, time always slowly steals our faces, and gradually some people will eventually leave us. The white cherry blossoms are pure and noble, the red cherry blossoms are passionate and unrestrained, and the green cherry blossoms are clear and elegant. The beauty and happiness of the flowers blooming, and the romance and elegance of the flowers falling all contain the life wisdom of cherry blossoms."

    data = {"project_name":"translation",
            "text":article_ar,
            "task":'en2zh'
            }
    encoded_data = json.dumps(data) #.encode("utf-8")

    

    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(json.loads(response))

async def tongchuan_en2zh():
    article_ar = "Sitting alone on a hill in the countryside"

    data = {"project_name":"translation",
            "text":article_ar,
            "task":'en2zh'
            }
    tts_data = {"project_name":"vits_tts_cn",
            "text":"",
            "gender":1,  #0 woman, 1 man
            }
    output_audio = "./temp/tongchuan_en2zh.wav"
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        tts_data['text'] = json.loads(response)['result']
    encoded_data = json.dumps(tts_data)
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        print(tts_data)
        await websocket.send(encoded_data)
        response = await websocket.recv()
        binary_data = base64.b64decode(json.loads(response)["audio_data"].encode())
        audio_array = np.frombuffer(binary_data, dtype=np.float32)
        write(output_audio, 16000, audio_array)


async def tongchuan_zh2en():
    article_hi = "独自坐在乡间的小丘上,看着阳光渐渐变暗,听着鸟鸣渐渐变弱,触着清风渐渐变凉"

    data = {"project_name":"translation",
            "text":article_hi,
            "task":'zh2en'
            }
    tts_data = {"project_name":"vits_tts_en",
            "text":"",
            "gender":1,  #0 woman, 1 man
            }
    output_audio = "./temp/tongchuan_zh2en.wav"
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        tts_data['text'] = json.loads(response)['result']
    encoded_data = json.dumps(tts_data)
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        binary_data = base64.b64decode(json.loads(response)["audio_data"].encode())
        audio_array = np.frombuffer(binary_data, dtype=np.float32)
        write(output_audio, 22050, audio_array)

def chat_with_bot(prompt:str)->dict:
        import urllib3
        data = {"prompt":prompt, "type":'101','stream':False}
        encoded_data = json.dumps(data).encode("utf-8")
        http1 = urllib3.PoolManager()
        r = http1.request('POST',"http://localhost:8501/inference",body=encoded_data)
        response = r.data.decode("utf-8")[5:].strip()
        output = json.loads(response)
        print(output)
        return output

def video_generation_offline():
    from myapp.video_generation_interface import Video_Generation_Interface
    from models.server_conf import ServerConfig
    global_conf = ServerConfig("./myapp/conf.yaml")
    model = Video_Generation_Interface(global_conf)
    
    data1 = {"project_name":"Video_Generation_Interface",
            #'prompt':'A car is running on the road.',
            'prompt':'A ugly monster is eating a person.',
            "num_frames":2,
            "if_sr":True,
            "ratio":'9:16',
            "image_start":"",
            "image_end":"",
            "continue_url":""
            }
    model.dsso_init(data1)
    response = model.dsso_forward(data1)
    print(response)

    from myapp.super_resulution_video import Super_Resolution_Video
    model = Super_Resolution_Video(global_conf)
    data = {"project_name":"super_resulution_video",
            "input_video":"samples/samples/sample_0000.mp4",
            'output_path':'samples/results/',
            }
    data["input_video"] = response['video']
    model.dsso_init(data)
    response,_ = model.dsso_forward(data)
    return response




async def video_generation_online():

    data = {"project_name":"video_generation",
            'prompt':'A car is running on the in forest.',
            "num_frames":4
            }
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(json.loads(response))

async def super_resolution_video():

    data = {"project_name":"super_resulution_video",
            "input_video":"samples/samples/sample_0000.mp4",
            'output_path':'samples/results/',
            }
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(json.loads(response))


async def video_generation_online2():


    data1 = {"project_name":"Video_Generation_Interface",
            'prompt':'A car is running on the road.',
            "num_frames":4,
            "if_sr":True,
            "image_start":"",
            "image_end":"",
            "ratio":0 #  16:9=0  9:16=1  1:1=2
            }
    output1 = {"video":"xxxx"}

    data2 = {"project_name":"super_resulution_video",
            "input_video":output1['video'],
            'output_path':'samples/results/',
            }
    output2 = {"video":"xxxx"}

    encoded_data1 = json.dumps(data1) #.encode("utf-8")   
    response = {} 
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data1)
        response = await websocket.recv()
        print(response)

    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        data = {"project_name":"super_resulution_video",
            "input_video":"samples/samples/sample_0000.mp4",
            'output_path':'samples/results/',
            }
        data["input_video"] = json.loads(response)['video']
        encoded_data2 = json.dumps(data) #.encode("utf-8")
        await websocket.send(encoded_data2)
        response = await websocket.recv()
        print(response)



async def video_generation_image2video():


    data1 = {"project_name":"Video_Generation_Interface",
            'prompt':'A girl is laughing.',
            "num_frames":4,
            "if_sr":True,
            "image_start":"/home/tione/notebook/lskong2/softwares1/Open-Sora-1.2.0/samples/samples/pictures/1.jpg",
            "image_end":"",
            "ratio":'16:9', #  16:9=0  9:16=1  1:1=2
            "continue_url":""
            }
    output1 = {"video":"xxxx"}

    data2 = {"project_name":"super_resulution_video",
            "input_video":output1['video'],
            'output_path':'samples/results/',
            }
    output2 = {"video":"xxxx"}

    encoded_data1 = json.dumps(data1) #.encode("utf-8")   
    response = {} 
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data1)
        response = await websocket.recv()
        print(response)

    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        data = {"project_name":"super_resulution_video",
            "input_video":"samples/samples/sample_0000.mp4",
            'output_path':'samples/results/',
            }
        data["input_video"] = json.loads(response)['video']
        encoded_data2 = json.dumps(data) #.encode("utf-8")
        await websocket.send(encoded_data2)
        response = await websocket.recv()
        print(response)


async def video_generation_connect():


    data1 = {"project_name":"Video_Generation_Interface",
            'prompt':'A girl is laughing.',
            "num_frames":4,
            "if_sr":True,
            "image_start":"samples/samples/pictures/s3.jpg",
            "image_end":"samples/samples/pictures/s4.jpg",
            "ratio":'16:9' #  16:9=0  9:16=1  1:1=2
            }
    output1 = {"video":"xxxx"}

    data2 = {"project_name":"super_resulution_video",
            "input_video":output1['video'],
            'output_path':'samples/results/',
            }
    output2 = {"video":"xxxx"}

    encoded_data1 = json.dumps(data1) #.encode("utf-8")   
    response = {} 
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data1)
        response = await websocket.recv()
        print(response)

    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        data = {"project_name":"super_resulution_video",
            "input_video":"samples/samples/sample_0000.mp4",
            'output_path':'samples/results/',
            }
        data["input_video"] = json.loads(response)['video']
        encoded_data2 = json.dumps(data) #.encode("utf-8")
        await websocket.send(encoded_data2)
        response = await websocket.recv()
        print(response)



def test_real_asr():
    import sys
    sys.path.append("/home/tione/notebook/lskong2/projects/2.tingjian/")
    import whisper
    import torch,torchaudio
    
    asr_model = whisper.load_model(
                name="small",
                download_root="../2.tingjian/models/"
                )

    file_name = "./temp/b.wav"
    waveform,sp = torchaudio.load(file_name,normalize=True)
    resampler = torchaudio.transforms.Resample(orig_freq=sp, new_freq=16000)
    waveform = resampler(waveform)
    result = asr_model.transcribe(waveform.squeeze(0))#, fp16=torch.cuda.is_available())
    text = result['text'].strip()
    print(text)


    with wave.open(file_name, "rb") as wf:
        framerate = wf.getframerate()
        frame = wf.getnframes()
        audio_data = wf.readframes(frame)
        audio_samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
        audio_tensor = torch.from_numpy(audio_samples).float()
        waveform = resampler(audio_tensor)
        result = asr_model.transcribe(waveform, fp16=torch.cuda.is_available())
        text = result['text'].strip()
        print(text)

def test3():
    a = 5
    if a>1:
        print("1")
    elif a>2:
        print("2")
    else:
        print("else")
    punctuation = set()
    punctuation.add('.')
    punctuation.add('!')
    punctuation.add('?')
    punctuation.add('。')
    punctuation.add('！')
    punctuation.add('？')
    pattern1 = f'[{"".join(re.escape(p) for p in punctuation)}]'

    sentence = "This is a test Is it working.. Greaet's continue"

    result = re.findall(rf'.+?{pattern1}', sentence)
    remaining_text = re.split(rf'{pattern1}', sentence)[-1]
    if remaining_text:
        result.append(remaining_text)

    print(result)


async def sam2():

    data = {"project_name":"sam2",
            "task_name":"test",
            'video':'/home/tione/notebook/lskong2/softwares1/segment-anything-2-main/videos/snake.mp4',
            "labels_points":[[261, 1252]],
            "labels":[1],
            "ann_frame_idx" : 0,  # the frame index we interact with
            "ann_obj_id" : 1
            }
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(json.loads(response))


async def sam1():

    data = {"project_name":"sam1",
            'image_url':'temp/20241009103740.jpg',
            }
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(json.loads(response))

def test_sam1():
    data = {"project_name":"sam1",
            'image_url':'temp/20241009103740.jpg',
            }
    from myapp.sam1 import Sam1
    from models.server_conf import ServerConfig
    global_conf = ServerConfig("./myapp/conf.yaml")
    model  = Sam1(global_conf)
    model.dsso_init()
    model.dsso_forward(data)


async def realtime_asr_en():
    import time
    start_time = time.time()
    async with websockets.connect(WS_URL,open_timeout=3000,close_timeout=3000) as websocket:

        # wf = wave.open('/home/tione/notebook/lskong2/projects/2.tingjian/test_set/tesla_autopilot.wav', "rb")
        # temp/1cdc7498c6d2b8dde71772e73e75af43.webm
        wf = open('./temp/tesla_autopilot.wav', "rb")

        sample_rate = 16000
        input = {"project_name": "realtime_asr_whisper",
                 "language_code": 'en', #zh 
                 "audio_data": None,  #和之前一样
                 "state": "continue",
                 "sample_rate": sample_rate,  # int
                 "translation_task":"none" # en2zh/zh2en/none  ##翻译任务，英到中/中到英/不翻译随便传
                 }
        
        output = {
            "key":
            list[
                dict[
                "output":str, ###转写结果字符串数组
                "trans":str | None, ###翻译结果字符串数组，可能有None
                "refactoring": bool, ###是否规整，如果True则转写结果深色展示，否则浅色展示
                "timestamp_start": float, ##句子起始时间
                "timestamp_end" :float ##句子结束时间
                ]
            ]
        }

        buffer_size = int(sample_rate * 2 * 1)  # 0.2 seconds of audio
        print("buffer_size: ", buffer_size)
        buffer = 0
        count = 0
        while True:
            """
            count +=1
            if count==120:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Elapsed time: {elapsed_time} seconds")
                # 107.0489194393158 seconds
                # without translation 91.54747581481934 seconds
                # without refactor 108
                exit(0)
            """
            data = wf.read(buffer_size)
            buffer += buffer_size
            print(buffer)
            if len(data) == 0:
                break
            encoded_audio = base64.b64encode(data).decode()
            input['audio_data'] = encoded_audio
            input['state'] = 'continue'

            await websocket.send(json.dumps(input))
            try:
                # Await a response from the WebSocket with a timeout
                response = await asyncio.wait_for(websocket.recv(), 1)
                print("Received response:", response)
            except asyncio.TimeoutError:
                print("Timeout: No response received within the specified time.")
            except websockets.ConnectionClosed:
                print("Connection closed")

        input['state'] = 'finished'
        await websocket.send(json.dumps(input))
        print(await websocket.recv())


def test_cos():
    from models.dsso_util import CosUploader
    uploader = CosUploader(0)

    response = uploader.upload_file("../../softwares1/segment-anything-main/models/sam_onnx_quantized_example.onnx")
    print(response)


async def realtime_asr_en_chatbot():
    import time
    start_time = time.time()
    async with websockets.connect(WS_URL,open_timeout=3000,close_timeout=3000) as websocket:

        # wf = wave.open('/home/tione/notebook/lskong2/projects/2.tingjian/test_set/tesla_autopilot.wav', "rb")
        # temp/1cdc7498c6d2b8dde71772e73e75af43.webm
        wf = open('./temp/tesla_autopilot.wav', "rb")

        sample_rate = 16000
        input = {"project_name": "realtime_asr_whisper_chatbot",
                 "language_code": 'en', #zh 
                 "audio_data": None,  #和之前一样
                 "state": "continue",
                 "sample_rate": sample_rate,  # int
                 "translation_task":"none" # en2zh/zh2en/none  ##翻译任务，英到中/中到英/不翻译随便传
                 }
        
        output = {
            "key":
            list[
                dict[
                "output":str, ###转写结果字符串数组
                "trans":str | None, ###翻译结果字符串数组，可能有None
                "refactoring": bool, ###是否规整，如果True则转写结果深色展示，否则浅色展示
                "timestamp_start": float, ##句子起始时间
                "timestamp_end" :float ##句子结束时间
                ]
            ]
        }

        buffer_size = int(sample_rate * 2 * 1)  # 0.2 seconds of audio
        print("buffer_size: ", buffer_size)
        buffer = 0
        count = 0
        while True:
            """
            count +=1
            if count==120:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Elapsed time: {elapsed_time} seconds")
                # 107.0489194393158 seconds
                # without translation 91.54747581481934 seconds
                # without refactor 108
                exit(0)
            """
            data = wf.read(buffer_size)
            buffer += buffer_size
            print(buffer)
            if len(data) == 0:
                break
            encoded_audio = base64.b64encode(data).decode()
            input['audio_data'] = encoded_audio
            input['state'] = 'continue'

            await websocket.send(json.dumps(input))
            try:
                # Await a response from the WebSocket with a timeout
                response = await asyncio.wait_for(websocket.recv(), 1)
                print("Received response:", response)
            except asyncio.TimeoutError:
                pass
            except websockets.ConnectionClosed:
                print("Connection closed")

        input['state'] = 'finished'
        await websocket.send(json.dumps(input))
        print(await websocket.recv())


async def online_asr_en_microphone():
    async with websockets.connect(WS_URL) as websocket:
        sample_rate = 16000
        input = {"project_name": "realtime_asr_whisper_chatbot",
                 "language_code": 'zh', #zh 
                 "audio_data": None,  #和之前一样
                 "sample_rate": sample_rate,  # int
                 "translation_task":"none" # en2zh/zh2en/none  ##翻译任务，英到中/中到英/不翻译随便传
                 }
        
        output = {"trans_text": "",  #转写结果
                   "response_text": "",  #大模型返回结果
                     "record": False,
                       "if_send": True,
                         "audio_length": 10.0,
                           "speech_timestamps": None,
                             "if_wait": True  #是否需要等待，此时不录音
                             }

        buffer_size = int(sample_rate)  # buffer size for 200ms
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=buffer_size)

        print("Recording...")
        count = 0
        try:
            while True:
                data = stream.read(buffer_size)
                encoded_audio = base64.b64encode(data).decode()
                input['audio_data'] = encoded_audio
                input['state'] = 'continue'

                await websocket.send(json.dumps(input))
                try:
                    # Await a response from the WebSocket with a timeout
                    response = await asyncio.wait_for(websocket.recv(), 1)
                    response = json.loads(response)
                    print("Received response1:", response)
                    while response['if_wait']:
                        response = await asyncio.wait_for(websocket.recv(), 10)
                        response = json.loads(response)
                        #print("Received response2:", response)
                        output_audio = "E:\\workspace\\code\\audio\\"+str(count)+".wav"
                        count+=1
                        binary_data = base64.b64decode(response["audio_data"].encode())
                        audio_array = np.frombuffer(binary_data, dtype=np.float32)
                        scipy.io.wavfile.write(output_audio, 16000, audio_array)
                except asyncio.TimeoutError:
                    pass
                except websockets.ConnectionClosed:
                    print("Connection closed")
        except KeyboardInterrupt:
            input['state'] = 'finished'
            await websocket.send(json.dumps(input))
            print(await websocket.recv())
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


async def motion_clone():

    data = {"project_name":"motion_clone","video_path":"temp/motion_clone/reference_videos/camera_zoom_out.mp4", "condition_image_paths":["temp/motion_clone/condition_images/rgb/dog_on_grass.png"], "new_prompt": "Dog, lying on the grass"}
    encoded_data = json.dumps(data) #.encode("utf-8")
    
    async with websockets.connect(WS_URL) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(f"Received from server: {response}")

async def jumper_cutter():

    data = {
        "project_name":"jumper_cutter",
        "video_path":"temp/jumpcutter-master/AnqXL8_886-pRwwf.mp4",
        "silent_speed":9999,
        "sounded_speed":1,
        }
    encoded_data = json.dumps(data) #.encode("utf-8")
    
    async with websockets.connect(WS_URL) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(f"Received from server: {response}")


async def fun_clip():

    data = {
        "project_name":"fun_clip",
        "input_video":"temp/funclip/boxing1.mp4",
        "language":"en",
        }
    encoded_data = json.dumps(data) #.encode("utf-8")
    
    async with websockets.connect(WS_URL) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(f"Received from server: {response}")


def test_local_motion_clone():
    from myapp.motion_clone import Motion_Clone
    from models.server_conf import ServerConfig
    from models.dsso_util import CosUploader
    conf_path = "./configs/conf.yaml"
    global_conf = ServerConfig(conf_path)
    input = {"video_path":"temp/motion_clone/reference_videos/camera_zoom_out.mp4", "condition_image_paths":["temp/motion_clone/condition_images/rgb/dog_on_grass.png"], "new_prompt": "Dog, lying on the grass"}
    model = Motion_Clone(
                    conf=global_conf,
                    uploader=CosUploader(global_conf.cos_uploader_mode),
                    executor=None,
                    time_blocker=global_conf.time_blocker,
                )
    model.dsso_forward(input)


def test_lang():
    import langid

    text = "你好"
    language, _ = langid.classify(text)

    print(f"Language: {language}")


def test_audiofromvideo():
    from moviepy.editor import VideoFileClip

    # Load the video file
    video = VideoFileClip("temp/funclip/boxing1.mp4")

    # Extract audio and save it as a .mp3 file
    video.audio.write_audiofile("temp/funclip/boxing1.wav")

if __name__ =="__main__":


    if len(sys.argv)<2:
        #test_audiofromvideo()
        #ai_meeting_chatbot_offline()
        asyncio.run(fun_clip())
        #vits_conversion()
    elif int(sys.argv[1]) == 1:
        asyncio.run(forgery())
    elif int(sys.argv[1]) == 2:
        asyncio.run(super_resolution())
    elif int(sys.argv[1]) == 3:
        asyncio.run(warning_detection1())
    elif int(sys.argv[1]) == 4:
        asyncio.run(ai_class())
    elif int(sys.argv[1]) == 5:
        asyncio.run(online_asr_en())
    elif int(sys.argv[1]) == 6:
        asyncio.run(online_asr_cn())
    elif int(sys.argv[1]) == 7:
        ##asyncio.run(ai_meeting())
        pass
    elif int(sys.argv[1]) == 8:
        asyncio.run(ai_meeting_chatbot())
    elif int(sys.argv[1]) ==9:   
        asyncio.run(vits_tts_en())  ###英文TTS
    elif int(sys.argv[1]) ==10:  
        asyncio.run(vits_tts_cn()) ###中文TTS
    elif int(sys.argv[1]) ==11: 
        asyncio.run(translation_zh2en())    ###中文-》英文  翻译
    elif int(sys.argv[1]) ==12: 
        asyncio.run(translation_en2zh())    ###英文-》中文  翻译
    elif int(sys.argv[1]) ==13:     
        asyncio.run(tongchuan_en2zh())  ###英文-》中文  同声传译
    elif int(sys.argv[1]) ==14:   
        asyncio.run(tongchuan_zh2en())  ###中文-》英文  同声传译
    elif int(sys.argv[1]) ==15:  
        asyncio.run(online_asr_en_webm())    ###麦克风实时语音识别 【英文实例】
    elif int(sys.argv[1]) ==16:
        chat_with_bot('您好')
    elif int(sys.argv[1]) ==17:
        asyncio.run(video_generation_online())
    elif int(sys.argv[1]) ==18:
        asyncio.run(super_resolution_video())
    elif int(sys.argv[1]) ==19:
        asyncio.run(video_generation_online2())
    elif int(sys.argv[1]) ==20:
        asyncio.run(video_generation_image2video())
    elif int(sys.argv[1]) ==21:
        asyncio.run(video_generation_connect())
    elif int(sys.argv[1]) ==22:
        asyncio.run(sam2())
    elif int(sys.argv[1]) ==23:
        asyncio.run(realtime_asr_en())
    elif int(sys.argv[1])== 24:
        asyncio.run(motion_clone())
    elif int(sys.argv[1]) == 25:
        asyncio.run(jumper_cutter())
    








