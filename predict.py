import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import cv2
from torch import IntTensor

#video_path = "./videos/st-catherines_drive.mp4"
video_path = "./videos/mcgill_drive.mp4"
cap = cv2.VideoCapture(video_path)
model = YOLO('./runs/detect/train9/weights/best.pt')
record={}
record[0]={}
record[1]={}
record[2]={}
count = [0,0,0]
frameparam = [60,40,10]
output_video = []
success, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=length-1,position=0, leave=True)
while cap.isOpened():

    success, img = cap.read()
    if success:
        gray = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        flow = np.array(cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 50, 3, 5, 1.2, 0))
        mix = np.concatenate((np.array(img),flow),axis=2)
        results = model.track(source=mix, verbose=False,persist=True, tracker='./bytetrack.yaml')
        for i,r in enumerate(results):
            for index, box in enumerate(r.boxes):
                if box.id is None: continue
                tracker_id = int(IntTensor.item(box.id[0]))
                classID = int(IntTensor.item(box.cls[0]))
                if tracker_id not in record[classID]:
                    record[classID][tracker_id]=1
                else:
                    record[classID][tracker_id]+=1
                    if record[classID][tracker_id]==frameparam[classID]:count[classID]+=1
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame,'moving car:'+str(count[0])+'\nparked car:'+str(count[1])+'\npedestrian:'+str(count[2]),(1000,500),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA )
        output_video.append(annotated_frame)
        pbar.update(1)
    else:
        break
pbar.close()
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
output = cv2.VideoWriter('mcgilldrive_count.mp4',fourcc , 30, (output_video[0].shape[1],output_video[0].shape[0]))
for frame in tqdm(output_video):
    output.write(frame)
output.release()
cap.release()