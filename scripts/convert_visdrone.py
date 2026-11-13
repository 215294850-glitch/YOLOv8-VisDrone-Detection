
from pathlib import Path
from PIL import Image
import shutil

# 项目根目录
ROOT = Path(r"D:\YOLOv8_VisDrone")
RAW_ROOT = ROOT / "datasets" / "raw"
OUT_ROOT = ROOT / "datasets" / "visdrone_yolo"

SPLITS = ["train", "val", "test-dev"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# VisDrone原始类别1~10映射为YOLO类别0~9
CLASS_MAP = {i: i - 1 for i in range(1, 11)}


def find_folder(root, name):
    """递归查找唯一的指定文件夹"""
    folders = [
        p for p in root.rglob("*")
        if p.is_dir() and p.name.lower() == name.lower()
    ]

    if len(folders) == 0:
        raise FileNotFoundError(
            f"没有找到文件夹：{name}，位置：{root}"
        )

    if len(folders) > 1:
        raise RuntimeError(
            f"发现多个{name}文件夹，请先检查目录：\n"
            + "\n".join(str(p) for p in folders)
        )

    return folders[0]


def convert_one_split(split):
    print("\n" + "=" * 60)
    print("开始转换：", split)
    print("=" * 60)

    source_root = RAW_ROOT / split

    images_dir = find_folder(source_root, "images")
    annotations_dir = find_folder(source_root, "annotations")

    output_images = OUT_ROOT / "images" / split
    output_labels = OUT_ROOT / "labels" / split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        p for p in images_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    converted = 0
    skipped_categories = 0
    invalid_boxes = 0
    empty_labels = 0

    for index, image_path in enumerate(image_files, 1):
        annotation_path = (
            annotations_dir / (image_path.stem + ".txt")
        )

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"图片缺少标注：{image_path.name}"
            )

        # 获取图片宽度和高度
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                f"图片尺寸异常：{image_path}"
            )

        yolo_lines = []

        with open(annotation_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split(",")

                if len(parts) < 8:
                    invalid_boxes += 1
                    continue

                try:
                    x, y, w, h = map(float, parts[:4])
                    category = int(float(parts[5]))
                except ValueError:
                    invalid_boxes += 1
                    continue

                # 仅保留10个有效检测类别
                # 0为ignored regions，11为others
                if category not in CLASS_MAP:
                    skipped_categories += 1
                    continue

                if w <= 0 or h <= 0:
                    invalid_boxes += 1
                    continue

                # 将边界框限制在图像范围内
                x1 = max(0.0, x)
                y1 = max(0.0, y)
                x2 = min(float(image_width), x + w)
                y2 = min(float(image_height), y + h)

                new_w = x2 - x1
                new_h = y2 - y1

                if new_w <= 0 or new_h <= 0:
                    invalid_boxes += 1
                    continue

                # 转换为YOLO归一化坐标
                x_center = (x1 + x2) / 2 / image_width
                y_center = (y1 + y2) / 2 / image_height
                width = new_w / image_width
                height = new_h / image_height

                class_id = CLASS_MAP[category]

                yolo_lines.append(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

        # 保存YOLO格式标签
        label_path = output_labels / (
            image_path.stem + ".txt"
        )

        with open(label_path, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")
            else:
                empty_labels += 1

        # 复制图片，不修改原始数据
        destination_image = output_images / image_path.name

        if not destination_image.exists():
            shutil.copy2(image_path, destination_image)

        converted += 1

        if index % 500 == 0:
            print(f"处理进度：{index}/{len(image_files)}")

    print("\n转换完成：", split)
    print("图片数量：", converted)
    print("有效类别外的标注框：", skipped_categories)
    print("无效标注框：", invalid_boxes)
    print("空标签图片数量：", empty_labels)
    print("输出图片目录：", output_images)
    print("输出标签目录：", output_labels)


if __name__ == "__main__":
    for split_name in SPLITS:
        convert_one_split(split_name)

    print("\n全部数据集转换完成！")
