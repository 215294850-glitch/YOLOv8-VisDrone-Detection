
from ultralytics import YOLO


def main():
    # 加载 YOLOv8n 基准模型
    model = YOLO("yolov8n.pt")

    # VisDrone 数据集训练配置
    model.train(
        data="configs/visdrone.yaml",
        epochs=150,
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
        name="baseline_reproduce"
    )


if __name__ == "__main__":
    main()
