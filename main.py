import torch
from pathlib import Path


model = torch.hub.load('/Users/themagendrans/Desktop/Yolo/yolov5', 'custom', path='/Users/themagendrans/Desktop/Yolo/yolov5/runs/train/exp3/weights/best.pt', source='local')
newclasses=['glass','cardboard','metal','plastic','styrofoam']
model.names = newclasses
img_path = '/Users/themagendrans/Desktop/Yolo/61c6jXbnbhL.jpg'

results = model(img_path)


results.show()


output_path = Path('output')
output_path.mkdir(parents=True, exist_ok=True)  
results.save(output_path)


print(results.pandas().xywh)
