from typing import Dict
from myapp.dsso_server import DSSO_SERVER
from myapp.server_conf import ServerConfig
import os
import torch
import torch.onnx
import torchaudio
torch.set_num_threads(1)
import os
import sys
sys.path.append("/home/tione/notebook/lskong2/projects/2.tingjian/")
import whisper
from tqdm import tqdm
print("loading AutoModelForCausalLM, AutoTokenizer ...")
from modelscope import AutoModelForCausalLM, AutoTokenizer
print("loading GenerationConfig ...")
from modelscope import GenerationConfig
from omegaconf import OmegaConf
from typing import Union
from collections import Counter

import json
import urllib.request
import logging
import shutil
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


@torch.no_grad()
class AI_Meeting(DSSO_SERVER):
    def __init__(self,conf:ServerConfig):
        super().__init__()
        self.conf = conf
        self._need_mem = self.conf.ai_meeting_mem
        self.global_result  = {}
        self.global_result['diarization_result']:dict[int:dict] = {}
        self.global_result['asr_result']:dict[int:dict] = {}
        self.global_result['summary_result']:dict[int:str] = {}
        self.global_result['summary_diarization_result']:dict[int:str] = []
        self.task_path = ""
        self.ori_wav = ""
        self.resampled_wav = ""
        self.concated_wav = ""
        self.rec_file = ""
        self.speakerlabels_:list[int] = []

    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        self._need_mem = self.conf.ai_meeting_mem
        self.task_father_path = self.conf.ai_meeting_temp_path+'/'
        self.min_combine_sents_sec_sample= self.conf.ai_meeting_min_combine_sents_sec*self.conf.ai_meeting_supported_sampling_rate

    def dsso_init(self,req:Dict = None)->bool:    
        pass

    def load_json(self,file_path:str)->dict:
        output  = {}
        if os.path.exists(file_path):
            with open(file_path) as fr:
                output = json.load(fr)
        converted_dict = {int(k): v for k, v in output.items()}
        return converted_dict
    
    def write_json(self,json_data:dict, file_path:str):
        json_data = json.dumps(json_data)
        with open(file_path, "w") as file:
             file.write(json_data)
        
    def audio_preprocess(
        self, 
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

    def get_speech_timestamps_silero_vad(
        self, 
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
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate_)
            return speech_timestamps

    def process_timestamps(self, timestamps_:list)->list:
        output_list = []
        for time_ in timestamps_:
            start_ = time_['start']
            end_ = time_['end']
            output_list.append([start_,end_])
        return output_list

    def trim_audio(
        self, 
        audio_:str,
        vad_list_:list[list[float]],
        min_combine_sents_sec_sample:int
        )->list[torch.tensor]:

        waveform,sr = torchaudio.load(audio_,normalize=True)
        current_sample = 0
        cut_waveform = torch.zeros([1, 0], dtype=torch.int32)
        output_tensors = []
        for s_e in vad_list_:
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
        return output_tensors

    def post_preprocess_asr_result(
        self,
        segments:list[dict],
        language_:str,
        )->list[str]:
        douhao = ', '
        juhao = '. '
        i = 0
        for i in range(0,len(segments)):
            if language_=='zh':
                if i+1<len(segments) and segments[i+1]['id']==0:
                    segments[i]['text']  = segments[i]['text']+juhao
                else:
                    segments[i]['text']  = segments[i]['text']+douhao
            else:
                pass
    
    def write_asr_result(
        self,
        segments:dict[int:dict],
        rec_file:str,
        transcribe_gap:int,
        text_encoding='utf8',
    ):
        f = open(rec_file,mode='w',encoding=text_encoding)
        last_gap = 0+transcribe_gap
        start = None
        end = 0
        transcribe_text = []
        #for segment in segments:
        for idx in range(0,len(segments)):
            segment = segments[idx]
            last_seg_duration = segment['last_seg_duration']/10
            end = segment['end']+last_seg_duration
            if start==None:
                start = segment['start']+last_seg_duration
                transcribe_text.append(segment['text'])
            elif end>last_gap:
                f.write('start: '+str(start)+' end: '+str(end)+'\n')
                f.write(''.join(transcribe_text))
                f.write('\n\n')
                last_gap += transcribe_gap
                start = segment['start']+last_seg_duration
                transcribe_text = []
                transcribe_text.append(segment['text'])
            else:
                transcribe_text.append(segment['text'])
        f.write('start: '+str(start)+' end: '+str(end)+'\n')
        f.write(''.join(transcribe_text))
        f.write('\n\n')
        f.close()

    def get_audio_duration(self, file_path):
    # Load the audio file
        waveform,sr = torchaudio.load(file_path,normalize=True)
        duration_seconds = round(waveform.shape[1]/sr)
        return duration_seconds

    def speaker_diarization_nemo(
        self,
        audio_file_:str,
        n_speakers_:int,
        manifest_file_in:str,
        manifest_file_out:str,
        diar_infer_meeting_file:str,
        output_dia:str,
        )->(list[int],dict[str,list]):

        labels = []
        speakerdict = {}
        secs = self.get_audio_duration(audio_file_)*10
        labels = [-1] * secs
        #cfg = "/home/tione/notebook/lskong2/projects/2.tingjian/NeMo-1.21.0/examples/speaker_tasks/diarization/conf/inference/diar_infer_meeting.yaml"
        #cfg = hydra_runner_get_cfg()
        
        cfg  = OmegaConf.load(diar_infer_meeting_file)
        f_json = open(manifest_file_in,mode='r')
        f_json_w = open(manifest_file_out,mode='w')
        json_dict = json.load(f_json)
        json_dict['audio_filepath'] = audio_file_
        json_dict['num_speakers'] = n_speakers_
        json.dump(json_dict,f_json_w)
        f_json.close()
        f_json_w.close()
        cfg['diarizer']['manifest_filepath'] = manifest_file_out
        cfg['diarizer']['out_dir'] = output_dia
        from nemo.collections.asr.models import ClusteringDiarizer
        sd_model = ClusteringDiarizer(cfg=cfg).to(cfg.device)
        sd_model.diarize()
        file_cluster = open(output_dia+'/speaker_outputs/subsegments_scale5_cluster.label',mode='r')
        lines = file_cluster.readlines()

        segment_list = []
        _,last_st,last_ed,sp = lines[0].strip().split()
        last_st = float(last_st)
        last_ed = float(last_ed)
        last_speaker = int(sp.split("_")[1])
        i = 1
        while i<len(lines):
            _,st,ed,sp = lines[i].strip().split()
            st = float(st)
            ed = float(ed)
            speaker = int(sp.split("_")[1])

            if speaker == last_speaker:
                last_st = min(last_st,st)
                last_ed = max(last_ed,ed)
            else:
                segment_list.append([last_speaker,last_st,last_ed])
                last_st = st
                last_ed = ed
                last_speaker = speaker
            i+=1

        segment_list.append([last_speaker,last_st,last_ed])
        for segment in segment_list:
            sp,st,ed = segment
            st = st*10
            ed = ed*10
            labels[round(st):round(ed)] = [sp]*(round(ed)-round(st))
        return labels,speakerdict
    
    def speaker_diarization_align_with_trans(
        self,
        speakerlabels:list[int],
        segments:list[dict]
        )->list[tuple]:
        # speakerlabels[1,1,3,4,5,6,]
        # speakerdict{'0':[[0,1],[2,3]],'1'[[4,5],[7,9]]}
        # segments  [{'text':str,'start':float,'end':'end'},{}]
        
        i = 0
        diarization_result:dict[int:dict] = {}

        #for segment in segments:
        count = 0
        for idx in range(0,len(segments)):
            segment = segments[idx]
            last_seg_duration = segment['last_seg_duration']
            text_ = segment['text']
            start_ = round((segment['start']*10+last_seg_duration))
            end_ = round((segment['end']*10+last_seg_duration))
            assert end_ < len(speakerlabels) , f"end_ should greater than len(labels), got: {end_,len(speakerlabels)}"
            if end_ > start_:
                counter = Counter(speakerlabels[start_:end_])
                speaker,_  = counter.most_common(1)[0]
                diarization_result[count] = {'speaker':speaker,'text':text_}
                count+=1  
        return diarization_result

    def transform_vadseg_to_audio(
        self,
        audio: Union[list[torch.tensor],torch.tensor],
        audio_file:str,
        sampling_rate_:int
        ):
        
        cut_waveform = torch.zeros([1, 0], dtype=torch.int32)
        if isinstance(audio, list) and len(audio)>1:
            for cut_waveform_ in audio:
                cut_waveform = torch.concat((cut_waveform,cut_waveform_),dim=1)
        torchaudio.save(audio_file, cut_waveform, sampling_rate_)

    def write_msdp_result(
        self,
        diarization_result:dict[int:dict],
        diar_file:str,
        text_encoding='utf8',
        ):
        last_speaker = None
        f = open(diar_file,mode='w',encoding=text_encoding)
        #for result in diarization_result:
        for idx in range(0,len(diarization_result)):
            result = diarization_result[idx]
            speaker,text = result["speaker"],result["text"]
            if last_speaker==None or last_speaker!=speaker:
                f.write('\n\nspeaker #{}\n{}\n'.format(speaker,text))
                last_speaker = speaker
            elif last_speaker == speaker:
                f.write(text+'\n')
        f.close()

    def meeting_summary(
        self,
        llm_path:str,
        max_tokens:int,
        language:str,
        prompt_cn_summarize:str,
        prompt_en_summarize:str,
        ):

        tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(llm_path, trust_remote_code=True) 
        def inner_check(tokenizer,input_str:str,max_tokens:int)->list:
            output = tokenizer.encode(input_str)
            return False if len(output)>=max_tokens else True

        max_lenths_rec_list = []
        def text_check(tokenizer,input_texts:list[dict],st:int,ed:int,max_tokens:int)->list:
            
            if inner_check(tokenizer,' '.join(input_texts[st:ed]),max_tokens):
                max_lenths_rec_list.append(' '.join(input_texts[st:ed]))
            else:
                mid = round((ed-st)/2+st)
                text_check(tokenizer,input_texts,st,mid,max_tokens)
                text_check(tokenizer,input_texts,mid,ed,max_tokens)

        textinputs = []
        for idx in range(0,len(self.global_result['asr_result'])):
            textinputs.append(self.global_result['asr_result'][idx]['text'])

        text_check(tokenizer,textinputs,0,len(textinputs),max_tokens)
        
        def llm_forward_with_check(model,prompt,history,tokenizer,st,ed,responses:list,idx:int):
            if inner_check(tokenizer,prompt[st:ed],max_tokens):
                logging.info("chatting with LLM...")
                response, _ = model.chat(tokenizer,prompt[st:ed] , history=history)
                responses.append([idx,response])
            else:
                mid = round((ed-st)/2+st)
                llm_forward_with_check(model,prompt,history,tokenizer,st,mid,responses,idx)
                llm_forward_with_check(model,prompt,history,tokenizer,mid,ed,responses,idx)
        i = 0
        article_text = ' '.join(max_lenths_rec_list)
        logging.info("meeting summary...")
        summary_result = []
        if language=='zh':
            prompt = prompt_cn_summarize+article_text
            llm_forward_with_check(
                model,
                prompt,
                None,
                tokenizer,
                0,
                len(prompt),
                summary_result,
                None
                )
        else: 
            prompt = prompt_en_summarize+article_text
            llm_forward_with_check(
                model,
                prompt,
                None,
                tokenizer,
                0,
                len(prompt),
                summary_result,
                None
                )
        

        for idx in range(0,len(summary_result)):
            self.global_result['summary_result'][idx] = summary_result[idx][1]
            
        
        i = 1
        last_speaker = self.global_result['diarization_result'][0]['speaker']
        last_article = self.global_result['diarization_result'][0]['text']
        diarization_summary_task = []
        while i< len(self.global_result['diarization_result']):

            current_speaker = self.global_result['diarization_result'][i]['speaker']
            current_article = self.global_result['diarization_result'][i]['text']

            if last_speaker == current_speaker:
                last_article+=current_article
            else:
                diarization_summary_task.append([last_speaker,last_article])
            i+=1
        if self.global_result['diarization_result'][len(self.global_result['diarization_result'])-1]['speaker']==last_speaker:
            diarization_summary_task.append([last_speaker,last_article])
        summary_diarization_result = []
        for current_speaker,current_article in diarization_summary_task:
            if language=='zh':
                prompt = prompt_cn_summarize+current_article
                llm_forward_with_check(
                    model,
                    prompt,
                    None,
                    tokenizer,
                    0,
                    len(prompt),
                    summary_diarization_result,
                    current_speaker
                    )
            else: 
                prompt = prompt_en_summarize+current_article
                llm_forward_with_check(
                    model,
                    prompt,
                    None,
                    tokenizer,
                    0,
                    len(prompt),
                    summary_diarization_result,
                    current_speaker
                    )
        for idx in range(0,len(summary_diarization_result)):
            self.global_result['summary_diarization_result'][idx] = summary_diarization_result[idx]

    def write_summary_result(
        self,
        output_encoding:str,
        summry_file:str,
        diar_sum_file:str,
    ):
        if self.conf.ai_meeting_if_write_summary:
            with open(summry_file,mode='w',encoding=output_encoding) as f:
                for idx in range(0,len(self.global_result['summary_result'])):
                    f.write(self.global_result['summary_result'][idx].strip()+'\n')

        if self.conf.ai_meeting_if_write_dia_summary:
            with open(diar_sum_file,mode='w',encoding=output_encoding) as f:
                for idx in range(0,len(self.global_result['summary_diarization_result'])):
                    speaker = self.global_result['summary_diarization_result'][idx][0]
                    text_ = self.global_result['summary_diarization_result'][idx][1]
                    f.write('speaker: num '+str(speaker)+'\n')
                    f.write(text_.strip()+'\n\n')


    def online_asr_process(self, request:Dict)->Dict:
        pass

    def offline_asr_process(self, request:Dict)->Dict:
        self.audio_preprocess(audio_file=self.ori_wav,
                     output_audio=self.resampled_wav,
                     ffmpeg_=self.conf.ai_meeting_ffmpeg_file,
                     sampling_rate_=self.conf.ai_meeting_supported_sampling_rate
                     )
        speech_timestamps = self.get_speech_timestamps_silero_vad(
                    audio_file=self.resampled_wav,
                    sampling_rate_=self.conf.ai_meeting_supported_sampling_rate,
                    vad_dir_=self.conf.ai_meeting_vad_dir
                    )
        speech_timestamps_list = self.process_timestamps(speech_timestamps)
        output_tensors = self.trim_audio(
                    audio_=self.resampled_wav,
                    vad_list_=speech_timestamps_list,
                    min_combine_sents_sec_sample=self.min_combine_sents_sec_sample
                    )
        
        output_tensors = output_tensors[0:min(len(output_tensors),self.conf.ai_meeting_max_decode)]

        if self.global_result["asr_result"] == None or len(self.global_result["asr_result"]) ==0:
            logging.info("Loading ASR model...")
            asr_model = whisper.load_model(
                    name=self.conf.ai_meeting_whisper_model_name,
                    download_root=self.conf.ai_meeting_asr_model_path
                    )

            logging.info("ASR Decoding...")
            results = []
            last_seg_duration = 0.0
            if self.conf.ai_meeting_if_greedy==1:
                self.conf.ai_meeting_beam_size = 1
            
            
            if self.conf.ai_meeting_language!="":
                idx = 0
                for i in tqdm(range(0,len(output_tensors))):
                    #output_tensors.append(cut_waveform.squeeze(0))
                    tensor_ = output_tensors[i].squeeze(0)
                    result = asr_model.transcribe(tensor_, word_timestamps=True,language=self.conf.ai_meeting_language,beam_size = self.conf.ai_meeting_beam_size)
                    results.append(result["text"])
                    result['segments'] = [{**d, 'last_seg_duration': last_seg_duration} for d in result['segments']]
                    for result in result['segments']:
                        self.global_result['asr_result'][idx] = result
                        idx+=1
                    #self.global_result['asr_result'].extend(result['segments'])
                    last_seg_duration += (tensor_.shape[0]/self.conf.ai_meeting_supported_sampling_rate*10)
                
            else:
                for i in tqdm(range(0,len(output_tensors))):
                    tensor_ = output_tensors[i].squeeze(0)
                    result = asr_model.transcribe(tensor_, word_timestamps=True,beam_size = self.conf.ai_meeting_beam_size)
                    results.append(result["text"])
                    result['segments'] = [{**d, 'last_seg_duration': last_seg_duration} for d in result['segments']]
                    #self.global_result['asr_result'].extend(result['segments'])
                    for result in result['segments']:
                        self.global_result['asr_result'][idx] = result
                        idx+=1
                    last_seg_duration += (tensor_.shape[0]/self.conf.ai_meeting_supported_sampling_rate*10)
            
            torch.cuda.empty_cache()
            logging.info("post_preprocess_asr_result...")
            self.post_preprocess_asr_result(
                segments=self.global_result['asr_result'],
                language_=self.conf.ai_meeting_language
                )
            
            #### write asr result to json file
            logging.info("dumping asr result...")
            self.write_json(self.global_result['asr_result'],self.rec_json)

            if self.conf.ai_meeting_if_write_asr:
                logging.info("Writing asr result...")
                self.write_asr_result(
                    segments=self.global_result['asr_result'],
                    rec_file=self.rec_file,
                    transcribe_gap=self.conf.ai_meeting_transcribe_gap,
                    text_encoding=self.conf.ai_meeting_output_encoding
                    )
            
        logging.info("transform_vadseg_to_audio...")
        self.transform_vadseg_to_audio(audio=output_tensors,
                                    audio_file=self.concated_wav,
                                    sampling_rate_=self.conf.ai_meeting_supported_sampling_rate)
        
        if self.global_result['diarization_result'] ==None or len(self.global_result['diarization_result'])==0:    
            if self.conf.ai_meeting_n_speakers>1 and self.conf.ai_meeting_if_msdp:
                logging.info("speaker_diarization_nemo...")
                print("loading ClusteringDiarizer ...")
                speakerlabels_,_ = self.speaker_diarization_nemo(
                    audio_file_=self.concated_wav,
                    n_speakers_=self.conf.ai_meeting_n_speakers,
                    manifest_file_in=self.conf.ai_meeting_manifest_file,
                    manifest_file_out=self.task_path+'/manifest_file_out.json',
                    diar_infer_meeting_file=self.conf.ai_meeting_diar_infer_meeting_cfg,
                    output_dia=self.task_path+'/output_dia/'
                    )

                logging.info("speaker_diarization_align_with_trans...")
                self.global_result['diarization_result'] = self.speaker_diarization_align_with_trans(
                speakerlabels=speakerlabels_,
                segments=self.global_result['asr_result'])

                #### write asr result to json file
                logging.info("dumping diarization result...")
                self.write_json(self.global_result['diarization_result'],self.diar_json)
            
            if self.conf.ai_meeting_if_write_msdp:
                logging.info("Writing msdp result...")
                self.write_msdp_result(
                    diarization_result=self.global_result['diarization_result'],
                    diar_file=self.diar_file)
            torch.cuda.empty_cache()
        if self.conf.ai_meeting_if_summary:
            logging.info("Extractiving Summarization...")
            self.meeting_summary(
                self.conf.ai_meeting_llm_path,
                self.conf.ai_meeting_max_tokens,
                self.conf.ai_meeting_language,
                self.conf.ai_meeting_prompt_cn_summarize,
                self.conf.ai_meeting_prompt_en_summarize
                )
            #### write asr result to json file
            logging.info("dumping sum result...")
            self.write_json(self.global_result['summary_result'],self.sum_json)
            logging.info("dumping sum diar result...")
            self.write_json(self.global_result['summary_diarization_result'],self.diar_sum_json)


            logging.info("Writing Summarization results...")
            self.write_summary_result(
                self.conf.ai_meeting_output_encoding,
                self.sum_file,
                self.diar_sum_file
                )

    def llm_summary(self, request:Dict)->Dict:
        pass


    def init_task(self,request:Dict):
        task_id  = str(request["task_id"]).strip()
        self.task_path  = self.task_father_path+task_id
        if len(task_id)>0:
            if os.path.exists(self.task_path):
                pass
            else:
                os.mkdir(self.task_path)
        self.ori_wav = self.task_path+'/'+'ori.wav'
        self.resampled_wav = self.task_path+'/'+'resampled.wav'
        self.concated_wav = self.task_path +'/' + 'concated.wav'
        if 'http' in request["audio_url"]:
            urllib.request.urlretrieve(request["audio_url"], self.ori_wav)
        else:
            shutil.copy(request["audio_url"], self.ori_wav)

        self.rec_file = self.task_path +'/' + 'rec.txt'
        self.diar_file = self.task_path +'/' + 'diar.txt'
        self.sum_file = self.task_path +'/' + 'sum.txt'
        self.diar_sum_file = self.task_path +'/' + 'diar_sum.txt'

        self.rec_json = self.task_path +'/' + 'rec.json'
        self.diar_json = self.task_path +'/' + 'diar.json'
        self.sum_json = self.task_path +'/' + 'sum.json'
        self.diar_sum_json = self.task_path +'/' + 'diar_sum.json'

        self.global_result['asr_result'] = self.load_json(self.rec_json)
        self.global_result['diarization_result'] = self.load_json(self.diar_json)
        self.global_result['summary_result']  = self.load_json(self.sum_json)
        self.global_result['summary_diarization_result']  = self.load_json(self.diar_sum_json)
    
    def dsso_forward(self, request: Dict) -> Dict:
        self.init_task(request)
        torch.cuda.set_device(self.conf.gpu_id)
        output_map = {}
        #request
        #{"task_id": "xxxx", "audio_url":"","task_state","", "task_type": ""}

        #task_type: online:0 / offline:1

        #task_state: start:0 / continue:1 / finish:2

        if request['task_type'] ==1 :
            output_map = self.offline_asr_process(request)
        elif request['task_type'] ==0 :
            output_map = self.online_asr_process(request)
        else :
            pass
        return output_map,True
