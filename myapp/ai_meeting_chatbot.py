from typing import Dict
from myapp.dsso_server import DSSO_SERVER
from models.server_conf import ServerConfig
from models.dsso_util import audio_preprocess,get_speech_timestamps_silero_vad,process_timestamps,trim_audio
import os
import torch
import torch.onnx
import torchaudio
import os
import sys
import urllib3
sys.path.append("./third_party/")
import whisper
from tqdm import tqdm
#from modelscope import  AutoTokenizer
from omegaconf import OmegaConf
from typing import Union
from collections import Counter
from models.dsso_util import CosUploader
import json
import urllib.request
import logging
import shutil
from models.dsso_model import DSSO_MODEL
from myapp.mbart_translation import mbart_translation
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


@torch.no_grad()
class AI_Meeting_Chatbot(DSSO_SERVER):
    def __init__(
        self,
        conf:ServerConfig,
        asr_model:DSSO_MODEL,
        translation_model:DSSO_MODEL,
        uploader:CosUploader
        ):
        print("--->initialize AI_Meeting_Chatbot...")
        super().__init__()
        self.conf = conf
        self._need_mem = self.conf.ai_meeting_mem
        self.global_result  = {}
        self.global_result['diarization_result']:dict[int:dict] = {}
        self.global_result['asr_result']:dict[int:dict] = {}
        self.global_result['summary_result'] = ""
        self.global_result['summary_diarization_result']:dict[int:str] = []
        self.task_path = ""
        self.ori_wav = ""
        self.resampled_wav = ""
        self.concated_wav = ""
        self.rec_file = ""
        self.speakerlabels_:list[int] = []
        self.task_father_path = self.conf.ai_meeting_temp_path+'/'
        self.min_combine_sents_sec_sample= self.conf.ai_meeting_min_combine_sents_sec*self.conf.ai_meeting_supported_sampling_rate
        self.uploader = uploader
        self.device = torch.device(self.conf.gpu_id)
        torch.cuda.set_device(self.conf.gpu_id)
        print("--->Loading ASR model...")
        self.mbart_translation_model = translation_model
        self.asr_model = asr_model

    def dsso_reload_conf(self,conf:ServerConfig):
        self.conf = conf
        
        self._need_mem = self.conf.ai_meeting_mem
        self.task_father_path = self.conf.ai_meeting_temp_path+'/'
        self.min_combine_sents_sec_sample= self.conf.ai_meeting_min_combine_sents_sec*self.conf.ai_meeting_supported_sampling_rate
        
        

    def dsso_init(self,req:Dict = None)->bool:
        if req["lang"] == "":
            self.conf.ai_meeting_language = ""
        else:
            self.conf.ai_meeting_language = req["lang"]
        
        if req["recognize_speakers"]==1:
            self.ai_meeting_n_speakers = 10
        else:
            self.ai_meeting_n_speakers = req["speaker_num"]

    def load_json(self,file_path:str)->dict:
        output  = {}
        if os.path.exists(file_path):
            with open(file_path) as fr:
                output = json.load(fr)
        if "/diar.json" in file_path or "/sum.json" in file_path:
            return output
        else:
            converted_dict = {int(k): v for k, v in output.items()}
            return converted_dict
    
    def chat_with_bot(self,prompt:str)->dict:
        try:
            data = {"prompt":prompt, "type":'101','stream':False}
            encoded_data = json.dumps(data).encode("utf-8")
            http1 = urllib3.PoolManager()
            r = http1.request('POST',self.conf.ai_meeting_chatbot_url,body=encoded_data)
            response = r.data.decode("utf-8")[5:].strip()
            output = json.loads(response)
        except Exception as e:
            print("LLM ERROR {}".format(e))
            output = 'LLM ERROR! '
        return output

    def chat_with_bot_timeout(self, prompt: str,count=0) -> dict:
        count+=1
        try:
            data = {"prompt": prompt, "type": '101', 'stream': False}
            encoded_data = json.dumps(data).encode("utf-8")
            http1 = urllib3.PoolManager()
            # Setting the timeout to 10 seconds
            r = http1.request(
                'POST',
                self.conf.ai_meeting_chatbot_url,
                body=encoded_data,
                timeout=30.0
            )
            response = r.data.decode("utf-8")[5:].strip()
            output = json.loads(response)
        except urllib3.exceptions.TimeoutError:
            if count >= 3:
                print("LLM ERROR: Request timed out. loop num {}".format(count))
                output = 'LLM ERROR: Request timed out.'
            else:
                return self.chat_with_bot_timeout(prompt,count)
        except Exception as e:
            if count >= 3:
                print("LLM ERROR: {}, loop num {}".format(e,count))
                output = 'LLM internal ERROR!.'
            else:
                return self.chat_with_bot_timeout(prompt,count)
        return output

    def write_json(self,json_data:dict, file_path:str):
        json_data = json.dumps(json_data)
        with open(file_path, "w") as file:
             file.write(json_data)

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
        cfg["diarizer"]["speaker_embeddings"]["model_path"]= "./third_party/NeMo_1.21.0/titanet-l/11ba0924fdf87c049e339adbf6899d48/titanet-l.nemo"
        cfg["diarizer"]["vad"]["model_path"]="./third_party/NeMo_1.21.0/vad_multilingual_marblenet/670f425c7f186060b7a7268ba6dfacb2/vad_multilingual_marblenet.nemo"
        from nemo.collections.asr.models import ClusteringDiarizer
        sd_model = ClusteringDiarizer(cfg=cfg).to(self.device)
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
        labels = [-1]*round(segment_list[-1][-1]*10)
        for segment in segment_list:
            sp,st,ed = segment
            st = st*10
            ed = ed*10
            labels[round(st):round(ed)] = [sp]*(round(ed)-round(st))
        return labels,speakerdict
    
    def post_diarization_result(
        self,
        diarization_result,
        ai_meeting_transcribe_gap
        )->list[tuple]:
        diarization_result_final:list[dict[int:dict]] = []

        if len(diarization_result)>1:
            last_speaker = diarization_result[0]['speaker']
            last_duration = diarization_result[0]['end'] - diarization_result[0]['start']
            last_text = diarization_result[0]['text']
            last_start = diarization_result[0]['start']
            last_end = diarization_result[0]['end']
        
        i = 1
        while i<len(diarization_result):
            c_speaker = diarization_result[i]['speaker']
            c_duration = diarization_result[i]['end'] - diarization_result[i]['start']
            c_text = diarization_result[i]['text']
            c_start = diarization_result[i]['start']
            c_end = diarization_result[i]['end']
            if last_speaker != c_speaker:
                if i == len(diarization_result)-1:
                    diarization_result_final.append({
                        'speaker':last_speaker,
                        'text':last_text,
                        'start':last_start,
                        'end':last_end,
                        })
                    diarization_result_final.append({
                    'speaker':c_speaker,
                    'text':c_text,
                    'start':c_start,
                    'end':c_end,
                    })
                    break
                else:
                    diarization_result_final.append({
                        'speaker':last_speaker,
                        'text':last_text,
                        'start':last_start,
                        'end':last_end,
                        })
                    last_speaker = c_speaker
                    last_duration = c_duration
                    last_text = c_text
                    last_start = c_start
                    last_end = c_end
                
            else:
                if i == len(diarization_result)-1:
                    diarization_result_final.append({
                        'speaker':c_speaker,
                        'text':last_text+c_text,
                        'start':last_start,
                        'end':c_end,
                        })
                else:
                    if last_duration + c_duration >ai_meeting_transcribe_gap:
                        diarization_result_final.append({
                        'speaker':last_speaker,
                        'text':last_text,
                        'start':last_start,
                        'end':last_end,
                        })
                        last_speaker = c_speaker
                        last_duration = c_duration
                        last_text = c_text
                        last_start = c_start
                        last_end = c_end
                    else:
                        last_duration += c_duration
                        last_text += c_text
                        last_end = c_end
            i+=1
        return diarization_result_final       
                
    def speaker_diarization_align_with_trans(
        self,
        speakerlabels:list[int],
        segments:list[dict],
        )->list[tuple]:
        # speakerlabels[1,1,3,4,5,6,]
        # speakerdict{'0':[[0,1],[2,3]],'1'[[4,5],[7,9]]}
        # segments  [{'text':str,'start':float,'end':'end'},{}]
        diarization_result:list[dict[int:dict]] = []

        #for segment in segments:
        speaker_dict = {}
        for idx in range(0,len(segments)):
            segment = segments[idx]
            last_seg_duration = segment['last_seg_duration']
            text_ = segment['text']
            start_ = round((segment['start']*10+last_seg_duration))
            end_ = round((segment['end']*10+last_seg_duration))
            if end_ > len(speakerlabels):
                hook = speakerlabels
                speakerlabels = [-1]*end_
                speakerlabels[0:len(hook)] = hook

            if end_ > start_:

                ### To find most common speaker in each segment.
                counter = Counter(speakerlabels[start_:end_])
                speaker = 0
                if len(counter.most_common())>1:
                    speaker1,_  = counter.most_common()[0]
                    speaker2,_  = counter.most_common()[1]
                    if speaker1 == -1:
                        speaker = speaker2
                    else:
                        speaker = speaker1
                elif len(counter.most_common())==1:
                    speaker1,_  = counter.most_common()[0]
                    if speaker1 == -1:
                        pass
                    else:
                        speaker = speaker1
                else:
                    pass
                ##done

                ## make sure each trans segment is aligned with the speaker segment
                if speaker in speaker_dict.keys():
                    speaker_ = speaker_dict[speaker]
                else:
                    speaker_dict[speaker] = len(speaker_dict)
                    speaker_ = speaker_dict[speaker]
                diarization_result.append({
                    'speaker':speaker_,
                    'text':text_,
                    'start':start_/10.0,
                    'end':end_/10.0,
                    })
                ##done

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
        diarization_result:list[dict[int:dict]],
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

        #tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        tokenizer = None
        def inner_check(tokenizer,input_str:str,max_tokens:int)->list:
            #output = tokenizer.encode(input_str)
            return False if 2*len(input_str)>=max_tokens else True

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
        
        def llm_forward_with_check(prompt,history,tokenizer,st,ed,responses:list,idx:int):
            if inner_check(tokenizer,prompt[st:ed],max_tokens):
                print("--->chatting with LLM...")
                try:
                    response = self.chat_with_bot_timeout(prompt[st:ed])["content"]
                except Exception as error:
                    print('--->LLM ERROR: ', error)
                    response = "LLM ERROR"
                responses.append([idx,response])
            else:
                mid = round((ed-st)/2+st)
                llm_forward_with_check(prompt,history,tokenizer,st,mid,responses,idx)
                llm_forward_with_check(prompt,history,tokenizer,mid,ed,responses,idx)
        i = 0
        article_text = ' '.join(max_lenths_rec_list)
        summary_result = []
        if language=='zh':
            prompt = prompt_cn_summarize+article_text
            llm_forward_with_check(
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
                prompt,
                None,
                tokenizer,
                0,
                len(prompt),
                summary_result,
                None
                )
        

        for idx in range(0,len(summary_result)):
            if self.conf.ai_meeting_language =="zh":
                self.global_result['summary_result']+=('第 '+str(idx+1)+' 部分:\n'+summary_result[idx][1]+'\n')
            else:
                self.global_result['summary_result']+=('No.'+str(idx+1)+'st part:\n'+summary_result[idx][1]+'\n')
            
        """
        i = 1
        last_speaker = self.global_result['diarization_result'][0]['speaker']
        last_article = self.global_result['diarization_result'][0]['text']
        """
        diarization_summary_task = {}
        i = 0
        while i< len(self.global_result['diarization_result']):
            """
            current_speaker = self.global_result['diarization_result'][i]['speaker']
            current_article = self.global_result['diarization_result'][i]['text']
            if last_speaker == current_speaker:
                last_article+=current_article
            else:
                diarization_summary_task.append([last_speaker,last_article])
                last_article = current_article
                last_speaker = current_speaker
            i+=1
            """
            c_speaker = self.global_result['diarization_result'][i]['speaker']
            c_article = self.global_result['diarization_result'][i]['text']+' '
            if c_speaker in diarization_summary_task.keys():
                diarization_summary_task[c_speaker]+=c_article
            else:
                diarization_summary_task[c_speaker] = c_article
            i+=1
        """
        if self.global_result['diarization_result'][len(self.global_result['diarization_result'])-1]['speaker']==last_speaker:
            diarization_summary_task.append([last_speaker,last_article])
        """
        summary_diarization_result = []
        #print("--->diarization_summary_task: ",diarization_summary_task)
        for current_speaker in range(0,len(diarization_summary_task)):
            current_article = diarization_summary_task[current_speaker]
            if language=='zh':
                prompt = prompt_cn_summarize+current_article
                llm_forward_with_check(
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
                    prompt,
                    None,
                    tokenizer,
                    0,
                    len(prompt),
                    summary_diarization_result,
                    current_speaker
                    )
        self.global_result['summary_diarization_result'] = {}
        for idx in range(0,len(summary_diarization_result)):
            c_speaker = summary_diarization_result[idx][0]
            c_article = summary_diarization_result[idx][1]+'\n'
            if c_speaker in self.global_result['summary_diarization_result'].keys():
                self.global_result['summary_diarization_result'][c_speaker]+=c_article
            else:
                self.global_result['summary_diarization_result'][c_speaker] = c_article

    def write_summary_result(
        self,
        output_encoding:str,
        summry_file:str,
        diar_sum_file:str,
    ):
        if self.conf.ai_meeting_if_write_summary:
            with open(summry_file,mode='w',encoding=output_encoding) as f:
                    f.write(self.global_result['summary_result'].strip())

        if self.conf.ai_meeting_if_write_dia_summary:
            with open(diar_sum_file,mode='w',encoding=output_encoding) as f:
                for speaker,text_ in self.global_result['summary_diarization_result'].items():
                    f.write('speaker: num '+str(speaker)+'\n')
                    f.write(text_.strip()+'\n\n')



    def online_asr_process(self, request:Dict)->Dict:
        pass

    def offline_asr_process(self, request:Dict)->Dict:

        if os.path.exists(self.resampled_wav) == False:
            print('--->audio_preprocess...')
            audio_preprocess(audio_file=self.ori_wav,
                        output_audio=self.resampled_wav,
                        ffmpeg_=self.conf.ai_meeting_ffmpeg_file,
                        sampling_rate_=self.conf.ai_meeting_supported_sampling_rate
                        )
        print('--->get_speech_timestamps_silero_vad...')
        speech_timestamps = get_speech_timestamps_silero_vad(
                    audio_file=self.resampled_wav,
                    sampling_rate_=self.conf.ai_meeting_supported_sampling_rate,
                    vad_dir_=self.conf.ai_meeting_vad_dir
                    )
        print('--->process_timestamps...')
        speech_timestamps_list = process_timestamps(speech_timestamps)
        print('--->trim_audio...')
        output_tensors = trim_audio(
                    audio_=self.resampled_wav,
                    vad_list_=speech_timestamps_list,
                    min_combine_sents_sec_sample=self.min_combine_sents_sec_sample
                    )
        
        output_tensors = output_tensors[0:min(len(output_tensors),self.conf.ai_meeting_max_decode)]

        if self.global_result["asr_result"] == None or len(self.global_result["asr_result"]) ==0:
            print("--->ASR Decoding...")
            results = []
            last_seg_duration = 0.0
            if self.conf.ai_meeting_if_greedy==1:
                self.conf.ai_meeting_beam_size = 1
            
            
            if self.conf.ai_meeting_language!="":
                idx = 0
                for i in tqdm(range(0,len(output_tensors))):
                    #output_tensors.append(cut_waveform.squeeze(0))
                    tensor_ = output_tensors[i].squeeze(0)
                    result = self.asr_model.predict_func_delay(
                        audio = tensor_,
                        word_timestamps=True,
                        language=self.conf.ai_meeting_language,
                        beam_size = self.conf.ai_meeting_beam_size
                    )

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
                    result = self.asr_model.predict_func_delay(
                        audio = tensor_,
                        word_timestamps=True,
                        beam_size = self.conf.ai_meeting_beam_size
                    )

                    results.append(result["text"])
                    result['segments'] = [{**d, 'last_seg_duration': last_seg_duration} for d in result['segments']]
                    #self.global_result['asr_result'].extend(result['segments'])
                    for result in result['segments']:
                        self.global_result['asr_result'][idx] = result
                        idx+=1
                    last_seg_duration += (tensor_.shape[0]/self.conf.ai_meeting_supported_sampling_rate*10)
            
            torch.cuda.empty_cache()
            print('--->post_preprocess_asr_result...')
            self.post_preprocess_asr_result(
                segments=self.global_result['asr_result'],
                language_=self.conf.ai_meeting_language
                )
            
            print('--->dumping asr result...')
            self.write_json(self.global_result['asr_result'],self.rec_json)

            if self.conf.ai_meeting_if_write_asr:
                print('--->Writing asr result...')
                self.write_asr_result(
                    segments=self.global_result['asr_result'],
                    rec_file=self.rec_file,
                    transcribe_gap=self.conf.ai_meeting_transcribe_gap,
                    text_encoding=self.conf.ai_meeting_output_encoding
                    )
            
        if not os.path.exists(self.concated_wav):
            if len(output_tensors) > 1:
                print('--->transform_vadseg_to_audio...')
                self.transform_vadseg_to_audio(audio=output_tensors,
                                        audio_file=self.concated_wav,
                                        sampling_rate_=self.conf.ai_meeting_supported_sampling_rate)
            elif len(output_tensors) ==1:
                shutil.copy(self.resampled_wav,self.concated_wav)
            else:
                pass


        self.global_result['concat_wav'] = self.uploader.upload_audio(self.concated_wav)
        if self.global_result['diarization_result'] ==None or len(self.global_result['diarization_result'])==0:    
            if self.conf.ai_meeting_n_speakers>1 and self.conf.ai_meeting_if_msdp:
                print("--->speaker_diarization_nemo...")
                speakerlabels_ = None
                try:
                    speakerlabels_,_ = self.speaker_diarization_nemo(
                        audio_file_=self.concated_wav,
                        n_speakers_=self.conf.ai_meeting_n_speakers,
                        manifest_file_in=self.conf.ai_meeting_manifest_file,
                        manifest_file_out=self.task_path+'/manifest_file_out.json',
                        diar_infer_meeting_file=self.conf.ai_meeting_diar_infer_meeting_cfg,
                        output_dia=self.task_path+'/output_dia/'
                        )
                except Exception as error:
                    self.global_result['state'] += (str(error).strip()+'\n')

                print("--->speaker_diarization_align_with_trans...")
                if speakerlabels_ == None:
                    pass
                else:
                    try:
                        self.global_result['diarization_result'] = self.speaker_diarization_align_with_trans(
                            speakerlabels = speakerlabels_,
                            segments = self.global_result['asr_result']
                            )
                    except Exception as error:
                        self.global_result['state'] += (str(error).strip()+'\n')
                print("--->post_diarization_result...")
                if len(self.global_result['diarization_result'])==0:
                    pass
                else:
                    try:
                        self.global_result['diarization_result'] = self.post_diarization_result(
                            self.global_result['diarization_result'],
                            ai_meeting_transcribe_gap = self.conf.ai_meeting_transcribe_gap
                            )
                    except Exception as error:
                        self.global_result['state'] += (str(error).strip()+'\n')

                print("--->dumping diarization result...")
                self.write_json(self.global_result['diarization_result'],self.diar_json)
            
            if self.conf.ai_meeting_if_write_msdp:
                print("--->write_msdp_result...")
                self.write_msdp_result(
                    diarization_result=self.global_result['diarization_result'],
                    diar_file=self.diar_file)
            torch.cuda.empty_cache()
        if self.conf.ai_meeting_if_summary:
            if self.global_result['summary_result'] =='' or len(self.global_result['summary_diarization_result'])==0:
                print("--->meeting_summary...")
                self.meeting_summary(
                    self.conf.ai_meeting_llm_path,
                    self.conf.ai_meeting_max_tokens,
                    self.conf.ai_meeting_language,
                    self.conf.ai_meeting_prompt_cn_summarize,
                    self.conf.ai_meeting_prompt_en_summarize
                    )
            print("--->dumping sum result...")
            self.write_json(self.global_result['summary_result'],self.sum_json)
            print("--->dumping sum diar result...")
            self.write_json(self.global_result['summary_diarization_result'],self.diar_sum_json)


            print("--->write_summary_result...")
            self.write_summary_result(
                self.conf.ai_meeting_output_encoding,
                self.sum_file,
                self.diar_sum_file
                )
            
        if request["trans"]==1:
            if request["lang"]=="en":
                request_trans = {}
                request_trans["task"] = 'en2zh'

                for i in range(0,len(self.global_result['diarization_result'])):

                    result = self.mbart_translation_model.predict_func_delay(
                        task = request_trans["task"],
                        text = self.global_result['diarization_result'][i]['text'])
                    
                    self.global_result['diarization_result'][i]["trans"] = result.strip()
                
            elif request["lang"]=="zh":
                request_trans = {}
                request_trans["task"] = 'zh2en'
                for i in range(0,len(self.global_result['diarization_result'])):

                    result = self.mbart_translation_model.predict_func_delay(
                        task = request_trans["task"],
                        text = self.global_result['diarization_result'][i]['text'])
                    self.global_result['diarization_result'][i]["trans"] = result.strip()
            else:
                pass

    def llm_summary(self, request:Dict)->Dict:
        pass


    def init_task(self,request:Dict):
        task_id  = str(request["task_id"]).replace(' ','').replace(':','_').replace('-','_').strip()
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
            if os.path.exists(self.ori_wav):
                pass
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

        if os.path.exists(self.rec_json):
            try:
                self.global_result['asr_result'] = self.load_json(self.rec_json)
            except Exception:
                self.global_result['asr_result'] =  {}
        else:
            self.global_result['asr_result'] =  {}

        if os.path.exists(self.diar_json):
            try:
                self.global_result['diarization_result'] = self.load_json(self.diar_json)
            except Exception:
                self.global_result['diarization_result'] =  {}
        else:
            self.global_result['diarization_result'] =  {}

        if os.path.exists(self.sum_json):
            try:
                self.global_result['summary_result']  = self.load_json(self.sum_json)
            except Exception:
                self.global_result['summary_result'] = ""
        else:
            self.global_result['summary_result'] = ""

        if os.path.exists(self.diar_sum_json):
            try:
                self.global_result['summary_diarization_result']  = self.load_json(self.diar_sum_json)
            except Exception:
                self.global_result['summary_diarization_result']  = {}
        else:
            self.global_result['summary_diarization_result'] = {}

        self.global_result['concat_wav'] = ""
        self.global_result['state'] = ""
    
    def dsso_forward(self, request: Dict) -> Dict:
        self.init_task(request)
        torch.cuda.set_device(self.conf.gpu_id)
        if request['task_type'] ==1 :
            self.offline_asr_process(request)
        elif request['task_type'] ==0 :
            self.online_asr_process(request)
        else :
            pass
        return self.global_result,True
