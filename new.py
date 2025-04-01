import torch 
model = torch.hub.load('/Users/themagendrans/Desktop/Yolo/yolov5', 'custom', path='/Users/themagendrans/Desktop/Yolo/yolov5/runs/train/exp3/weights/best.pt', source='local')
 
print(model.names)
