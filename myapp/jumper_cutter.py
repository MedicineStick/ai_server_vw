

from typing import Dict
from myapp.dsso_server import DSSO_SERVER
from models.server_conf import ServerConfig
from models.dsso_util import CosUploader
import concurrent.futures.thread
import asyncio
import subprocess
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import copyfile, rmtree
import os
from pytube import YouTube

def downloadFile(url):
    name = YouTube(url).streams.first().download()
    newname = name.replace(' ','_')
    os.rename(name,newname)
    return newname

def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

def copyFrame(inputFrame,outputFrame,jumpcutter_temp_floder):
    src = jumpcutter_temp_floder+"/frame{:06d}".format(inputFrame+1)+".jpg"
    dst = jumpcutter_temp_floder+"/newFrame{:06d}".format(outputFrame+1)+".jpg"
    if not os.path.isfile(src):
        return False
    copyfile(src, dst)
    if outputFrame%20 == 19:
        print(str(outputFrame+1)+" time-altered frames saved.")
    return True

def inputToOutputFilename(filename):
    dotIndex = filename.rfind(".")
    return filename[:dotIndex]+"_ALTERED"+filename[dotIndex:]

def createPath(s):
    #assert (not os.path.exists(s)), "The filepath "+s+" already exists. Don't want to overwrite it. Aborting."

    try:  
        os.mkdir(s)
    except OSError:  
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"

def deletePath(s): # Dangerous! Watch out!
    try:  
        rmtree(s,ignore_errors=False)
    except OSError:  
        print ("Deletion of the directory %s failed" % s)
        print(OSError)


class ARGS():
    def __init__(self) -> None:
        self.input_file = ""
        self.url = None
        self.output_file = ""
        self.silent_threshold = 0.3
        self.sounded_speed = 1
        self.silent_speed = 999999
        self.frame_margin = 1
        self.sample_rate = 44100
        self.frame_rate = 30
        self.frame_quality = 3

def cutter_exec(
    input_path:str,
    output_path:str,  
    jumpcutter_temp_floder:str, 
    sounded_speed:int,
    silent_speed:int,
    )->str:
    """
    parser = argparse.ArgumentParser(description='Modifies a video file to play at different speeds when there is sound vs. silence.')
    parser.add_argument('--input_file', type=str,  help='the video file you want modified')
    parser.add_argument('--url', type=str, help='A youtube url to download and process')
    parser.add_argument('--output_file', type=str, default="", help="the output file. (optional. if not included, it'll just modify the input file name)")
    parser.add_argument('--silent_threshold', type=float, default=0.03, help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". It ranges from 0 (silence) to 1 (max volume)")
    parser.add_argument('--sounded_speed', type=float, default=1.00, help="the speed that sounded (spoken) frames should be played at. Typically 1.")
    parser.add_argument('--silent_speed', type=float, default=5.00, help="the speed that silent frames should be played at. 999999 for jumpcutting.")
    parser.add_argument('--frame_margin', type=float, default=1, help="some silent frames adjacent to sounded frames are included to provide context. How many frames on either the side of speech should be included? That's this variable.")
    parser.add_argument('--sample_rate', type=float, default=44100, help="sample rate of the input and output videos")
    parser.add_argument('--frame_rate', type=float, default=30, help="frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.")
    parser.add_argument('--frame_quality', type=int, default=3, help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 is the default.")
    args = parser.parse_args()
    """

    args = ARGS()
    args.input_file = input_path
    args.output_file = output_path
    args.silent_speed = silent_speed
    args.sounded_speed = sounded_speed
    frameRate = args.frame_rate
    SAMPLE_RATE = args.sample_rate
    SILENT_THRESHOLD = args.silent_threshold
    FRAME_SPREADAGE = args.frame_margin
    NEW_SPEED = [args.silent_speed, args.sounded_speed]
    if args.url != None:
        INPUT_FILE = downloadFile(args.url)
    else:
        INPUT_FILE = args.input_file
    URL = args.url
    FRAME_QUALITY = args.frame_quality

    assert INPUT_FILE != None , "why u put no input file, that dum"
        
    if len(args.output_file) >= 1:
        OUTPUT_FILE = args.output_file
    else:
        OUTPUT_FILE = inputToOutputFilename(INPUT_FILE)

    TEMP_FOLDER = jumpcutter_temp_floder
    AUDIO_FADE_ENVELOPE_SIZE = 400 # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)
        
    createPath(TEMP_FOLDER)

    command = "ffmpeg -i "+INPUT_FILE+" -qscale:v "+str(FRAME_QUALITY)+" "+TEMP_FOLDER+"/frame%06d.jpg -hide_banner"
    subprocess.call(command, shell=True)

    command = "ffmpeg -i "+INPUT_FILE+" -ab 160k -ac 2 -ar "+str(SAMPLE_RATE)+" -vn "+TEMP_FOLDER+"/audio.wav"

    subprocess.call(command, shell=True)

    command = "ffmpeg -i "+TEMP_FOLDER+"/input.mp4 2>&1"
    f = open(TEMP_FOLDER+"/params.txt", "w")
    subprocess.call(command, shell=True, stdout=f)



    sampleRate, audioData = wavfile.read(TEMP_FOLDER+"/audio.wav")
    audioSampleCount = audioData.shape[0]
    maxAudioVolume = getMaxVolume(audioData)

    f = open(TEMP_FOLDER+"/params.txt", 'r+')
    pre_params = f.read()
    f.close()
    params = pre_params.split('\n')
    for line in params:
        m = re.search('Stream #.*Video.* ([0-9]*) fps',line)
        if m is not None:
            frameRate = float(m.group(1))

    samplesPerFrame = sampleRate/frameRate

    audioFrameCount = int(math.ceil(audioSampleCount/samplesPerFrame))

    hasLoudAudio = np.zeros((audioFrameCount))



    for i in range(audioFrameCount):
        start = int(i*samplesPerFrame)
        end = min(int((i+1)*samplesPerFrame),audioSampleCount)
        audiochunks = audioData[start:end]
        maxchunksVolume = float(getMaxVolume(audiochunks))/maxAudioVolume
        if maxchunksVolume >= SILENT_THRESHOLD:
            hasLoudAudio[i] = 1

    chunks = [[0,0,0]]
    shouldIncludeFrame = np.zeros((audioFrameCount))
    for i in range(audioFrameCount):
        start = int(max(0,i-FRAME_SPREADAGE))
        end = int(min(audioFrameCount,i+1+FRAME_SPREADAGE))
        shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
        if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]): # Did we flip?
            chunks.append([chunks[-1][1],i,shouldIncludeFrame[i-1]])

    chunks.append([chunks[-1][1],audioFrameCount,shouldIncludeFrame[i-1]])
    chunks = chunks[1:]

    outputAudioData = np.zeros((0,audioData.shape[1]))
    outputPointer = 0

    lastExistingFrame = None
    for chunk in chunks:
        audioChunk = audioData[int(chunk[0]*samplesPerFrame):int(chunk[1]*samplesPerFrame)]
        
        sFile = TEMP_FOLDER+"/tempStart.wav"
        eFile = TEMP_FOLDER+"/tempEnd.wav"
        wavfile.write(sFile,SAMPLE_RATE,audioChunk)
        with WavReader(sFile) as reader:
            with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
                tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
                tsm.run(reader, writer)
        _, alteredAudioData = wavfile.read(eFile)
        leng = alteredAudioData.shape[0]
        endPointer = outputPointer+leng
        outputAudioData = np.concatenate((outputAudioData,alteredAudioData/maxAudioVolume))

        #outputAudioData[outputPointer:endPointer] = alteredAudioData/maxAudioVolume

        # smooth out transitiion's audio by quickly fading in/out
        
        if leng < AUDIO_FADE_ENVELOPE_SIZE:
            outputAudioData[outputPointer:endPointer] = 0 # audio is less than 0.01 sec, let's just remove it.
        else:
            premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE)/AUDIO_FADE_ENVELOPE_SIZE
            mask = np.repeat(premask[:, np.newaxis],2,axis=1) # make the fade-envelope mask stereo
            outputAudioData[outputPointer:outputPointer+AUDIO_FADE_ENVELOPE_SIZE] *= mask
            outputAudioData[endPointer-AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1-mask

        startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame))
        endOutputFrame = int(math.ceil(endPointer/samplesPerFrame))
        for outputFrame in range(startOutputFrame, endOutputFrame):
            inputFrame = int(chunk[0]+NEW_SPEED[int(chunk[2])]*(outputFrame-startOutputFrame))
            didItWork = copyFrame(inputFrame,outputFrame,TEMP_FOLDER)
            if didItWork:
                lastExistingFrame = inputFrame
            else:
                copyFrame(lastExistingFrame,outputFrame,TEMP_FOLDER)

        outputPointer = endPointer

    wavfile.write(TEMP_FOLDER+"/audioNew.wav",SAMPLE_RATE,outputAudioData)

    '''
    outputFrame = math.ceil(outputPointer/samplesPerFrame)
    for endGap in range(outputFrame,audioFrameCount):
        copyFrame(int(audioSampleCount/samplesPerFrame)-1,endGap)
    '''

    command = "ffmpeg -framerate "+str(frameRate)+" -i "+TEMP_FOLDER+"/newFrame%06d.jpg -i "+TEMP_FOLDER+"/audioNew.wav -strict -2 "+OUTPUT_FILE
    subprocess.call(command, shell=True)

    deletePath(TEMP_FOLDER)




class Jumper_Cutter(DSSO_SERVER):
    def __init__(self,
                 conf:ServerConfig,
                 uploader:CosUploader,
                 executor:concurrent.futures.thread.ThreadPoolExecutor,
                 time_blocker:int
                 ):
        print("--->initialize Jumper_Cutter...")
        super().__init__(time_blocker=time_blocker)
        self.uploader = uploader
        self.conf  = conf
        self.executor = executor

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
        output_video = self.conf.jumpcutter_output+request["task_id"]+".mp4"
        output_map["output"] = cutter_exec(
            input_path=request["video_path"],
            output_path=output_video,
            jumpcutter_temp_floder=self.conf.jumpcutter_temp_floder,
            sounded_speed=self.conf.jumpcutter_sounded_speed,
            silent_speed=self.conf.jumpcutter_silent_speed,
            )
        output_map["output"] = self.uploader.upload_video(output_video)
        output_map['state'] = 'finished'
        return output_map
