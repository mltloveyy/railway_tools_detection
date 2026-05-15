import json
import os
import shutil

import cv2
import numpy as np
import requests
import yaml


class BboxAnnotation:
    def __init__(
        self,
        root_dirs: list,
        yaml_path: str,
        save_dir: str,
        ignores: list = None,
    ):
        self.classes = {}
        self.root_dirs = root_dirs
        self.save_dir = save_dir
        self.ignores = ignores
        self.bing_api_key = ""
        self.search_endpoint = "https://api.bing.microsoft.com/v7.0/images/search"
        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        for type, subtypes in data.items():
            for key, info in subtypes.items():
                if key in ignores:
                    print(f"Ignore {key}")
                    continue
                self.classes[key] = dict(id=info["id"], name=info["name"], type=type, count=0)

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
        Convert rect to yolo format(cx, cy, w, h).
        """
        x_min = min([p[0] for p in points])
        x_max = max([p[0] for p in points])
        y_min = min([p[1] for p in points])
        y_max = max([p[1] for p in points])

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min

        dw = 1.0 / imgsz[0]
        dh = 1.0 / imgsz[1]
        cx = cx * dw
        cy = cy * dh
        w = w * dw
        h = h * dh
        return (cx, cy, w, h)

    def count(self, check: bool = False):
        """
        Count every category number or check annotation patches.
        """
        for root_dir in self.root_dirs:
            if check:
                check_dir = root_dir + "_check"
                if os.path.exists(check_dir):
                    shutil.rmtree(check_dir)
            for filename in os.listdir(root_dir):
                if not filename.endswith(".json"):
                    continue
                json_path = os.path.join(root_dir, filename)
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                image = None
                if check:
                    image_path = os.path.join(root_dir, data["imagePath"])
                    with open(image_path, "rb") as stream:
                        bytes = bytearray(stream.read())
                        numpy_array = np.asarray(bytes, dtype=np.uint8)
                        image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

                for i, shape in enumerate(data["shapes"]):
                    category = shape["label"]
                    if category not in self.classes.keys():
                        print(f"Unknow category {category} in {json_path}")
                        continue
                    self.classes[category]["count"] += 1
                    if check:
                        category_dir = os.path.join(check_dir, category)
                        os.makedirs(category_dir, exist_ok=True)
                        imgsz = (data["imageWidth"], data["imageHeight"])
                        x_min, y_min, x_max, y_max = self.coco2xyxy(shape["points"], imgsz)
                        patch = image[y_min:y_max, x_min:x_max]
                        patch_path = os.path.join(category_dir, f"{os.path.splitext(filename)[0]}_{i}.png")
                        cv2.imwrite(patch_path, patch)

    def download_images(self, download_dir: str, category, count):
        """
        Download images from Bing API
        """
        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
        params = {
            "q": category,
            "license": "public",
            "imageType": "photo",
            "count": min(count, 150),
            "safeSearch": "Strict",
        }

        try:
            response = requests.get(self.search_endpoint, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()

            save_dir = os.path.join(download_dir, category)
            os.makedirs(save_dir, exist_ok=True)

            downloaded = 0
            for idx, item in enumerate(search_results.get("value", [])):
                if downloaded >= count:
                    break

                image_url = item["contentUrl"]
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()

                image_data = np.frombuffer(response.content, dtype=np.uint8)
                img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"Skipped invalid image: {image_url}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                save_path = os.path.join(save_dir, f"web_{idx}.jpg")
                cv2.imwrite(save_path, img_rgb)
                downloaded += 1

            print(f"Successfully downloaded {downloaded}/{count} images of '{category}'")

        except Exception as e:
            print(f"Search request failed for '{category}': {str(e)}")

    def generate_config(self):
        """
        Save config.yaml.
        """
        sorted_classes = sorted(self.classes.items(), key=lambda x: x[1]["id"])
        names_dict = {}
        for name, info in sorted_classes:
            names_dict[info["id"]] = name

        yaml_data = dict(
            path=os.path.abspath(self.save_dir),
            train="train",
            val="val",
            test="test",
            names=names_dict,
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
        dataset_dir = os.path.join(self.save_dir, "train")
        os.makedirs(dataset_dir, exist_ok=True)
        for root_dir in self.root_dirs:
            for filename in os.listdir(root_dir):
                if not filename.endswith(".json"):
                    continue
                json_path = os.path.join(root_dir, filename)
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if len(data["shapes"]) == 0:
                    continue

                txt_path = os.path.join(dataset_dir, os.path.splitext(filename)[0] + ".txt")
                with open(txt_path, "w", encoding="utf-8") as out_file:
                    for shape in data["shapes"]:
                        category = shape["label"]
                        if category not in self.classes.keys():
                            print(f"Unknow category {category} in {json_path}")
                            continue
                        index = self.classes[category]["id"]
                        imgsz = (data["imageWidth"], data["imageHeight"])
                        xywh = self.coco2yolo(shape["points"], imgsz)
                        out_file.write(f"{index} {' '.join(map(str, xywh))}\n")

                if not os.path.exists(os.path.join(dataset_dir, data["imagePath"])):
                    shutil.copy(os.path.join(root_dir, data["imagePath"]), dataset_dir)


if __name__ == "__main__":
    root_dirs = [
        r"D:\code\detection\datasets\20241115",
        r"D:\code\detection\datasets\20241207",
        r"D:\code\detection\datasets\20250522",
    ]
    yaml_path = r"D:\code\detection\datasets\railwaytools.yaml"
    save_dir = r"D:\code\detection\datasets\trainsets\20250528"
    ignores = []
    anno = BboxAnnotation(root_dirs, yaml_path, save_dir, ignores)
    anno.count(check=False)
    for k, v in anno.classes.items():
        id = v["id"]
        c = v["count"]
        print(f"{k} id: {id}, count: {c}")

    anno.generate_config()
    anno.to_dataset()
