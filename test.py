import torch
import cv2

model = torch.hub.load('/Users/themagendrans/Desktop/Yolo/yolov5', 'custom', path='/Users/themagendrans/Desktop/Yolo/yolov5/runs/train/exp3/weights/best.pt', source='local')


video_path = 0  
cap = cv2.VideoCapture(video_path)


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


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()