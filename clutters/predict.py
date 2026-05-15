from ultralytics import YOLO

# Load a model
model = YOLO("Yolo/runs/detect/train5/weights/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(
    [
        "/home/yy/workspace/datasets/railwaytools/val/20241115191908.jpg",
        "/home/yy/workspace/datasets/railwaytools/val/20241115191906.jpg",
    ],
    imgsz=1024,
    conf=0.3,
)  # return a list of Results objects

# Process results list
for i, r in enumerate(results):
    boxes = r.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    r.show()  # display to screen
    r.save(filename=f"result_{i}.jpg")  # save to disk
