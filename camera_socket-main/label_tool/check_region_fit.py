"""区域适配检查工具：对一批图片按指定方案（区域表）裁剪出检测窗口，拼成一张总览图。

用途：
1. 标定新方案的区域框时快速迭代（一眼看出框是否套住螺钉）
2. 上线前 / 日常抽检：确认每张图的螺钉都完整落在区域内（防机械漂移）

本工具不需要模型和 GPU，纯裁剪拼接。总览图按文件名排序、每行 6 格排列。

用法（在项目根目录或 label_tool 下运行均可）：
    python label_tool/check_region_fit.py 图片文件夹 [方案号] [输出图路径]
方案号默认 0，输出图默认 label_tool/region_fit_overview.jpg。
"""
import os
import sys
import cv2
import numpy as np

# 把项目根目录加入模块搜索路径（本脚本在 label_tool 子目录中，regions.py 在项目根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regions import load_screw_regions, crop_screw_regions

# 总览图排版：每格宽度（像素）、每行格数、格间距
THUMB_WIDTH = 320
COLUMNS = 6
GAP = 4
# 总格数上限（防多区域方案拼出超大图），超过则只展示前面的
MAX_TILES = 200
# 默认输出图路径
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "region_fit_overview.jpg")
# 支持的图片扩展名
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']


def imread_unicode(file_path):
    """读取图片（兼容中文路径）"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(file_path, image):
    """写图片（兼容中文路径）"""
    ext = os.path.splitext(file_path)[1] or '.jpg'
    success, buffer = cv2.imencode(ext, image)
    if success:
        buffer.tofile(file_path)
    return success


def main():
    if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
        print("用法：python label_tool/check_region_fit.py 图片文件夹 [方案号] [输出图路径]")
        sys.exit(1)
    image_dir = sys.argv[1]
    scheme = 0
    if len(sys.argv) > 2:
        try:
            scheme = int(sys.argv[2])
        except ValueError:
            print(f"方案号 '{sys.argv[2]}' 不是整数")
            sys.exit(1)
    output_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT

    config = load_screw_regions(scheme=scheme)
    if config is None:
        print("错误：未找到有效的区域表方案")
        sys.exit(1)

    image_files = sorted(
        name for name in os.listdir(image_dir)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    )
    if not image_files:
        print(f"文件夹中没有图片：{image_dir}")
        sys.exit(1)

    tiles = []
    processed = 0
    read_fail = 0
    capped = False
    for name in image_files:
        image = imread_unicode(os.path.join(image_dir, name))
        if image is None:
            print(f"读取失败，跳过：{name}")
            read_fail += 1
            continue
        processed += 1
        for region, crop, _, _ in crop_screw_regions(image, config):
            scale = THUMB_WIDTH / crop.shape[1]
            thumb = cv2.resize(crop, (THUMB_WIDTH, max(1, int(round(crop.shape[0] * scale)))))
            tiles.append(thumb)
            if len(tiles) >= MAX_TILES:
                capped = True
                break
        if capped:
            break

    if not tiles:
        print("没有生成任何裁剪窗口（检查区域框是否都在图像范围内）")
        sys.exit(1)

    thumb_height = max(tile.shape[0] for tile in tiles)
    rows = (len(tiles) + COLUMNS - 1) // COLUMNS
    canvas = np.full((rows * (thumb_height + GAP) + GAP, COLUMNS * (THUMB_WIDTH + GAP) + GAP, 3),
                     60, dtype=np.uint8)
    for index, thumb in enumerate(tiles):
        row, col = divmod(index, COLUMNS)
        x = GAP + col * (THUMB_WIDTH + GAP)
        y = GAP + row * (thumb_height + GAP)
        canvas[y:y + thumb.shape[0], x:x + THUMB_WIDTH] = thumb

    if not imwrite_unicode(output_path, canvas):
        print(f"总览图写入失败：{output_path}")
        sys.exit(1)

    print(f"完成：处理 {processed} 张图（读取失败 {read_fail} 张） | 拼出 {len(tiles)} 个裁剪窗口"
          + ("（已达上限，仅展示前面的）" if capped else ""))
    print(f"总览图已保存：{output_path}")
    print("请目检：每格的螺钉是否都完整落在画面内（画面就是模型的全部视野）")


if __name__ == "__main__":
    main()
