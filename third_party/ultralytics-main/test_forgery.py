from ultralytics import YOLO
import torch
from PIL import Image
from ultralytics.utils.plotting import Annotator 
torch.cuda.set_device(1)
model = YOLO('/home/tione/notebook/lskong2/projects/3.forgery_detection/ultralytics-main/runs/detect_v2/train2/weights/best.pt')  # load an official model
torch.cuda.set_device(1)
model.to('cuda:1')

threshold = 0.1


with open("/home/tione/notebook/lskong2/projects/3.forgery_detection/data/VOC2024_2/test/images.list",mode='r') as f1:
      
    lines = f1.readlines()

    total = len(lines)
    count = 0
    for  im1 in lines:
        im2 = im1.strip()
        im2 = Image.open(im2)
        results = model.predict(source=im2, save=False)
        forgery = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf.cpu().numpy()[0]>threshold:
                    forgery = True
                    count+=1
                    b = box.xyxy[0]
                    print(im1)
                    break
                    

    print(count,total)
exit(0)
img = "/home/tione/notebook/lskong2/projects/3.forgery_detection/data/VOC2024_2/test/images/1088.png"
results = model.predict(source=img, save=False)
for r in results:
    boxes = r.boxes
    for box in boxes:
        if box.conf.cpu().numpy()[0]>threshold:
            b = box.xyxy[0]
        print(box)





        



# Export the model
#results = model("/home/tione/notebook/lskong2/projects/3.forgery_detection/data/original_images/0.invoice/11.jpg")

#print(results)