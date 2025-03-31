import torch
from pathlib import Path

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img_path = 'imgpath.jpg'

results = model(img_path)

results.show()


output_path = Path('output')
output_path.mkdir(parents=True, exist_ok=True)  
results.save(output_path)


print(results.pandas().xywh)