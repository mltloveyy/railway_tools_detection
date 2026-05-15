import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

img_list = [
    "/home/yy/workspace/datasets/railwaytools/val/20241115191908.jpg",
    "/home/yy/workspace/datasets/railwaytools/val/20241115191906.jpg",
]

# Load a model
model = YOLO("/home/yy/workspace/detection/Yolo/runs/detect/train5/weights/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(
    img_list,
    imgsz=1024,
    conf=0.3,
)

# Create SAMPredictor
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    imgsz=1024,
    model=r"D:\code\detection\Yolo\models\sam_b.pt",
    save=False,
)
predictor = SAMPredictor(overrides=overrides)

# Process results list
for i, r in enumerate(results):
    img_show = cv2.imread(img_list[i], cv2.IMREAD_UNCHANGED)
    predictor.set_image(img_show)
    boxes = r.boxes
    for box in boxes:
        label = r.names[int(box.cls)]
        rect = box.xyxy[0].tolist()

        sam_results = predictor(bboxes=rect, points=None)

        cv2.rectangle(img_show, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), 2)
        mask_pts = sam_results[0].masks.xy[0].astype(np.int32)

        cv2.polylines(img_show, [mask_pts], True, [0, 0, 255], 2)
        # cv2.fillPoly(img_show, [mask_pts], [0, 0, 255])

        print(f"label: {label}, rect: {rect}")
    cv2.imshow("img", img_show)
    cv2.waitKey(0)
