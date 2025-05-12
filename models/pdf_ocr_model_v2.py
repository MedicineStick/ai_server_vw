

from models.dsso_model import DSSO_MODEL
from models.server_conf import ServerConfig
import os, subprocess
from pdf2image import convert_from_path
from PIL import Image
import urllib
import shutil

class PDF_OCR_Model_V2(DSSO_MODEL):
    def __init__(self, conf:ServerConfig):
        super().__init__(time_blocker=conf.time_blocker)
        self.if_available = True
        self.conf = conf
    
    def pdf_to_html_preview(self,pdf_path, html_path, image_dir="./temp/ocr"):
        os.makedirs(image_dir, exist_ok=True)

        # Convert PDF pages to images
        images = convert_from_path(pdf_path)

        # Save each page as an image
        image_paths = []
        for i, img in enumerate(images):
            img_path = os.path.join(image_dir, f"page_{i+1}.png")
            img.save(img_path, "PNG")
            image_paths.append(img_path)

        # Generate HTML content
        html = "<html><head><meta charset='utf-8'><title>PDF Preview</title></head><body>\n"
        for path in image_paths:
            html += f"<img src='{path}' style='width:100%; margin-bottom:20px;'>\n"
        html += "</body></html>"

        # Save HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Preview saved to {html_path}")

    def predict_func(self, **kwargs)->dict:
        # audio, word_timestamps, language, beam_size,initial_prompt,fp16
        input_image = kwargs["image_url"]
        img_path = "./temp/ocr/"
        img_name = input_image[input_image.rfind('/') + 1:]
        pdf_path = os.path.join(img_path,img_name)
        html_path = os.path.join(img_path,img_name[:img_name.rfind('.')]+".html")
        if 'http' in input_image:
                urllib.request.urlretrieve(input_image,pdf_path)
        else:
            if os.path.exists(pdf_path):
                pass
            else:
                shutil.copy(input_image, pdf_path)
        
        self.pdf_to_html_preview(pdf_path, html_path)
        return {"result": html_path}

        
        
        
        
        

        