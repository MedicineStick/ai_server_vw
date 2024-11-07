

from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig
import json
import urllib3

class DssoLLM(DSSO_MODEL):
    def __init__(self, conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        self.if_available = True
        self.conf = conf

        
    def predict_func(self, **kwargs)->dict:
        prompt = kwargs["prompt"]
        retry_count = int(kwargs["retry_count"])
        timeout = float(kwargs["timeout"])
        count = int(kwargs["count"])
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
                timeout=timeout
            )
            response = r.data.decode("utf-8")[5:].strip()
            output = json.loads(response)["content"]
        except urllib3.exceptions.TimeoutError:
            if count >= retry_count:
                print("LLM ERROR: Request timed out. loop num {}".format(count))
                output = 'LLM ERROR: Request timed out.'
            else:
                return self.predict_func(
                    prompt = prompt,
                    retry_count = retry_count,
                    timeout = timeout,
                    count = count,
                    )
        except Exception as e:
            if count >= retry_count:
                print("LLM ERROR: {}, loop num {}".format(e,count))
                output = 'LLM internal ERROR!.'
            else:
                return self.predict_func(
                    prompt = prompt,
                    retry_count = retry_count,
                    timeout = timeout,
                    count = count,
                    )
        return output


        