from ultralytics import SAM

# Load a model
model = SAM("Yolo/sam_b.pt")

# Display model information (optional)
model.info()

# # Run inference with bboxes prompt
# results = model("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])

# # Run inference with single point
# results = model(points=[900, 370], labels=[1])

# # Run inference with multiple points
# results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

# # Run inference with multiple points prompt per object
# results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

# # Run inference with negative points prompt
# results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])

# Run inference with whole image
results = model("/home/yy/workspace/datasets/railwaytools/val/20241115191906.jpg")

# Process results list
for i, r in enumerate(results):
    boxes = r.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    r.show()  # display to screen
    r.save(filename=f"result_{i}.jpg")  # save to disk

print("OK!")
