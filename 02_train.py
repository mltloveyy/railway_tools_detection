import argparse
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--config", type=str, default="data/datasets/20250531/config.yaml", help="The path to the dataset configuration file")
    parser.add_argument("--model", type=str, default="models/yolo26x.pt", help="The model file for training")
    parser.add_argument("--model_type", type=str, default="yolo26x.yaml", help="The model type for training")
    parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="Device to run training on")
    parser.add_argument("--multi_scale", type=float, default=0.25, help="Randomly vary imgsz each batch by +/-")
    parser.add_argument("--export_mnn", action="store_true", help="Export to MNN format")
    parser.add_argument("--precision", type=str, default="fp32", help="Export precision. Options: fp32, fp16, int8")
    args = parser.parse_args()

    # Load a pretrained model
    if args.train:
        model = YOLO(args.model_type).load(args.model)

        # Train
        results = model.train(
            data=args.dataset,
            epochs=args.epoch,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            save_period=10,
            # amp=False,
            multi_scale=args.multi_scale,
        )

        # Evaluate
        metrics = model.val(data=args.dataset)
    else:
        model = YOLO(args.model)

    # Export to MNN format
    if args.export_mnn:
        if args.precision == "fp16":
            model.export(format="mnn", imgsz=args.imgsz, half=True)
        elif args.precision == "int8":
            model.export(format="mnn", imgsz=args.imgsz, int8=True)
        else:
            model.export(format="mnn", imgsz=args.imgsz)
