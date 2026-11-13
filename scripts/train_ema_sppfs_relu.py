
from ultralytics import YOLO


def main():
    # 加载包含 EMA 和 SPPF-S 的模型配置
    model = YOLO("configs/yolov8n_ema_sppfs.yaml")

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
        name="ema_sppfs_relu_100_reproduce"
    )


if __name__ == "__main__":
    main()
