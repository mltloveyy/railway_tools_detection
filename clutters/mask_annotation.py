import json
import os
import shutil

import cv2
import numpy as np
import yaml
from ultralytics.models.sam import Predictor as SAMPredictor


class MaskAnnotation:
    def __init__(
        self,
        root_dirs: list,
        save_dir: str,
        model_path: str,
        yaml_path: str,
        imgsz: int = 1024,
    ):
        self.root_dirs = root_dirs
        self.save_dir = save_dir
        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        self.names = {}
        for _, subtypes in data.items():
            for key, info in subtypes.items():
                self.names[info["id"]] = key
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=imgsz, model=model_path, save=False)
        self.predictor = SAMPredictor(overrides=overrides)

    @staticmethod
    def coco2xyxy(points: list, imgsz: tuple):
        """
        Convert rect to (x_min, y_min, x_max, y_max).
        """
        x_min = min([p[0] for p in points])
        x_max = max([p[0] for p in points])
        y_min = min([p[1] for p in points])
        y_max = max([p[1] for p in points])

        x_max = min(x_max, imgsz[0])
        y_max = min(y_max, imgsz[1])
        return (x_min, y_min, x_max, y_max)

    @staticmethod
    def coco2yolo(points: list, imgsz: tuple):
        """
        Convert points to yolo format(x/dw, y/dh).
        """
        points_np = np.array(points)
        factors = np.array([1.0 / imgsz[0], 1.0 / imgsz[1]])
        points_yolo = (points_np * factors).flatten().tolist()
        return points_yolo

    @staticmethod
    def mask2polygons(mask: np.ndarray):
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
            epsilon = 2e-3 * cv2.arcLength(contour, True)
            pts = cv2.approxPolyDP(contour, epsilon, True)
            if pts.size < 6:
                continue  # less than 6 points, skip
            polygons.append(pts.squeeze())
        return polygons

    def bbox2mask(self):
        self.seg_root_dirs = []
        for root_dir in self.root_dirs:
            save_dir = root_dir + "_seg"
            self.seg_root_dirs.append(save_dir)
            for filename in os.listdir(root_dir):
                if not filename.endswith(".json"):
                    continue
                json_path = os.path.join(root_dir, filename)
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if len(data["shapes"]) == 0:
                    continue

                image_path = os.path.join(root_dir, data["imagePath"])
                with open(image_path, "rb") as stream:
                    bytes = bytearray(stream.read())
                    numpy_array = np.asarray(bytes, dtype=np.uint8)
                    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
                self.predictor.set_image(image)

                seg_data = dict(
                    version=data["version"],
                    flags=data["flags"],
                    shapes=[],
                    imagePath=data["imagePath"],
                    imageData=None,
                    imageHeight=data["imageHeight"],
                    imageWidth=data["imageWidth"],
                )
                for i, shape in enumerate(data["shapes"]):
                    if shape["shape_type"] != "rectangle":
                        continue
                    imgsz = (data["imageWidth"], data["imageHeight"])
                    xyxy = self.coco2xyxy(shape["points"], imgsz)
                    results = self.predictor(bboxes=xyxy, points=None)
                    mask = results[0].masks.data.cpu().numpy().squeeze(0).astype(np.uint8) * 255
                    polygons = self.mask2polygons(mask)
                    for poly in polygons:
                        seg_data["shapes"].append(
                            dict(
                                label=shape["label"],
                                points=[[float(p[0]), float(p[1])] for p in poly],
                                shape_type="polygon",
                                group_id=i,
                                description="",
                                flags={},
                                mask=None,
                            )
                        )

                os.makedirs(save_dir, exist_ok=True)
                json_savepath = os.path.join(save_dir, filename)
                with open(json_savepath, "w", encoding="utf-8") as f:
                    json.dump(seg_data, f, ensure_ascii=False, indent=2)
                if not os.path.exists(os.path.join(save_dir, data["imagePath"])):
                    shutil.copy(image_path, save_dir)

    def generate_config(self):
        """
        Save config.yaml.
        """
        yaml_data = dict(
            path=os.path.abspath(self.save_dir),
            train="train",
            val="val",
            test="test",
            names=self.names,
        )
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, "config.yaml")
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            print(f"config.yaml save at {self.save_dir}")

    def to_dataset(self):
        """
        Generate dataset.
        """
        for root_dir in self.seg_root_dirs:
            for filename in os.listdir(root_dir):
                if not filename.endswith(".json"):
                    continue
                json_path = os.path.join(root_dir, filename)
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if len(data["shapes"]) == 0:
                    continue

                group_polygons = {}
                for shape in data["shapes"]:
                    category = shape["label"]
                    index = next((k for k, v in self.names.items() if v == category), None)
                    if index is None:
                        print(f"Unknow category {category} in {json_path}")
                        continue
                    group_id = shape["group_id"]
                    group_polygons[group_id] = dict(id=index, points=[])
                    group_polygons[group_id]["points"].extend(shape["points"])

                txt_path = os.path.join(self.save_dir, os.path.splitext(filename)[0] + ".txt")
                with open(txt_path, "w", encoding="utf-8") as out_file:
                    for group_id, info in group_polygons.items():
                        index = info["id"]
                        imgsz = (data["imageWidth"], data["imageHeight"])
                        points = self.coco2yolo(info["points"], imgsz)
                        out_file.write(f"{index} {' '.join(map(str, points))}\n")

                if not os.path.exists(os.path.join(self.save_dir, data["imagePath"])):
                    shutil.copy(os.path.join(root_dir, data["imagePath"]), self.save_dir)


if __name__ == "__main__":
    root_dirs = [
        r"D:\code\detection\datasets\20241115",
        r"D:\code\detection\datasets\20241207",
        r"D:\code\detection\datasets\20250522",
    ]
    save_dir = r"D:\code\detection\datasets\trainsets\20250528_seg"
    model_path = r"D:\code\detection\Yolo\models\mobile_sam.pt"
    yaml_path = r"D:\code\detection\datasets\railwaytools.yaml"

    anno = MaskAnnotation(root_dirs, save_dir, model_path, yaml_path)

    anno.bbox2mask()

    # anno.generate_config()

    # anno.to_dataset()
