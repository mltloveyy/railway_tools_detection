import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

img_path = r"D:\code\detection\dataset\20241207\0b7792ac48734a3aa796579943cfae65.jpg"

# Create SAMPredictor
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    imgsz=1024,
    model=r"Yolo\models\sam_b.pt",
    save=False,
)
predictor = SAMPredictor(overrides=overrides)

img_show = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
predictor.set_image(img_show)

# point annotation
# points = [[[340, 370], [400, 370], [400, 450]]]
# labels = [[1, 1, 1]]
# for i in range(len(points[0])):
#     cv2.circle(img_show, (points[0][i][0], points[0][i][1]), 5, (0, 255, 0), -1)

# results = predictor(bboxes=None, points=points, labels=labels)[0].masks

# rect annotation
rect = [92, 243, 696, 456]
results = predictor(bboxes=rect, points=None)[0].masks

#
pts = results.xy[0].astype(np.int32)  # points of mask
mask = results.data.cpu().numpy().squeeze(0).astype(np.uint8) * 255  # mask image

cv2.polylines(img_show, [pts], True, [0, 0, 255], 2)
# cv2.fillPoly(img_show, [mask_pts], [0, 0, 255])

box = cv2.boundingRect(pts)
cv2.rectangle(img_show, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

cv2.imshow("img", img_show)
cv2.waitKey(0)
