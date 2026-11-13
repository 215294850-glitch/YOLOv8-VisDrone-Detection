
from ultralytics import YOLO


def main():
    model = YOLO("configs/yolov8n_ema.yaml")

    model.train(
        data="configs/visdrone.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        workers=2,
        device=0,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        cos_lr=True,
        seed=42,
        project="runs/detect",
        name="ema_reproduce"
    )


if __name__ == "__main__":
    main()
