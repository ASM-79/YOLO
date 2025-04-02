import torch
import cv2
import pygame

model = torch.hub.load('/Users/themagendrans/Desktop/Yolo/yolov5', 'custom', path='/Users/themagendrans/Desktop/Yolo/yolov5/runs/train/exp3/weights/last.pt', source='local')
newclasses=['glass','cardboard','metal','plastic','styrofoam']
model.names = newclasses

video_path = 0  
cap = cv2.VideoCapture(video_path)
print (cap.get(cv2.CAP_PROP_FPS))



if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
   
    results = model(rgb_frame)
    
    
    
    img = results.render()[0]  
    

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('YOLOv5 Video Detection', img_bgr)
    if results.pandas().xyxy[0].empty:
        waittime=0
        pass
    else:
        waittime+=1
        if waittime==30:
            results.show()
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()