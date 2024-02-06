from ultralytics import YOLO
from glob import glob
from PIL import Image, ImageDraw
import re

model = YOLO('./runs/detect/train6/weights/best.pt')

image_files = glob('datasets/test/images/*.jpeg')
image_files = [file for file in image_files if re.match(r'datasets/test/images/\d+\.jpeg$', file)]

results = model.predict(image_files)

for result in results:
    for index in range(len(result.boxes.conf)):
        print(f'Index: {index+1}')
        x, y = float(result.boxes.xywh[index][0]), float(result.boxes.xywh[index][1])
        
        print(result.boxes.conf[index])
        print(x, y)
        
        img_array = result.plot()  # plot a BGR numpy array of predictions
        img = Image.fromarray(img_array[..., ::-1])  # RGB PIL imgage
        
        draw = ImageDraw.Draw(img)
        
        # 원의 반지름을 정합니다.
        r = 3

        # 원을 그립니다. fill 파라미터는 원의 색상을 지정합니다.
        draw.ellipse((x-r, y-r, x+r, y+r), fill='red')
        
        img.show()  # show image
    
    print('\n------------------\n')