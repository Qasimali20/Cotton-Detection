from ultralytics import YOLO
import cv2
from collections import defaultdict

model = YOLO("runs/detect/train/weights/best.pt")  

class_names = ['bud', 'cotton', 'flower']


image_path = "test/images/20240926_174128_jpg.rf.a507a72f1bff569b7575cf9fbac0934e.jpg"  # Replace with your image path
image = cv2.imread(image_path)
results = model(image)

result = results[0]

counts = defaultdict(int)
for box in result.boxes:
    class_id = int(box.cls[0])
    counts[class_names[class_id]] += 1


print("Object counts:")
for name in class_names:
    print(f"{name}: {counts[name]}")


annotated_frame = result.plot() 
cv2.imshow("Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
