
from pathlib import Path

# VisDrone原始数据集根目录
ROOT = Path(r"D:\YOLOv8_VisDrone\datasets\raw")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_folder(root, name):
    """递归查找指定名称的文件夹"""
    folders = [
        p for p in root.rglob("*")
        if p.is_dir() and p.name.lower() == name.lower()
    ]

    if not folders:
        raise FileNotFoundError(
            f"没有找到 {name} 文件夹：{root}"
        )

    if len(folders) > 1:
        print(f"警告：发现多个 {name} 文件夹：")
        for p in folders:
            print(" ", p)

    return folders[0]


def check_split(split):
    print("\n" + "=" * 55)
    print(f"检查数据集：{split}")
    print("=" * 55)

    split_root = ROOT / split

    images_dir = find_folder(split_root, "images")
    labels_dir = find_folder(split_root, "annotations")

    images = [
        p for p in images_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = [
        p for p in labels_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
    ]

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    matched = image_stems & label_stems
    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    print("图片目录：", images_dir)
    print("标注目录：", labels_dir)
    print("图片数量：", len(images))
    print("标注文件数量：", len(labels))
    print("文件名匹配数量：", len(matched))
    print("缺少标注的图片：", len(missing_labels))
    print("缺少图片的标注：", len(missing_images))

    if missing_labels:
        print("缺少标注的图片示例：",
              list(missing_labels)[:5])

    if missing_images:
        print("缺少图片的标注示例：",
              list(missing_images)[:5])


if __name__ == "__main__":
    for split in ["train", "val", "test-dev"]:
        check_split(split)

    print("\n数据集检查完成！")
