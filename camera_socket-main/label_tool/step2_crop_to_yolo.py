"""第二步：根据人工审核后的 labelme json 和 stud_detect_regions.json 区域表裁剪训练小图。

裁剪窗口 = 区域表中的矩形框（ENLARGE_RATIO=0 时不做外扩，即人工框本身）。
每个窗口内最多一个钉：窗口内可见面积 ≥ MIN_VISIBLE_RATIO 的标注框若有多个，
说明区域表配置不对，取可见面积最大的一个并打印警告；
一个都没有则生成空 txt 作为负样本（用于空位照片，供"判无钉"训练）；
例外：单区域方案下 json 里有框但都不在窗口内属"区域框/标注不一致"，警告并跳过该样本。
每张有 json 的图固定生成与区域数相同的样本数（正样本 + 负样本）。

用法（在项目根目录运行）：
    python label_tool/step2_crop_to_yolo.py 图片文件夹路径 输出文件夹路径 [区域表方案号]
方案号不传则默认 0。

输出结构（ultralytics 标准约定，训练 yaml 直接指向 images 即可）：
    输出文件夹/images/  裁出的小图，命名 {原图名}_r{区域id}.jpg
    输出文件夹/labels/  同名 .txt（空文件 = 负样本）
"""
import os
import sys
import json
import cv2
import numpy as np

# 把项目根目录加入模块搜索路径（本脚本在 label_tool 子目录中，regions.py 在项目根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regions import load_screw_regions

# 裁剪窗口：区域矩形框四周外扩的比例。0 = 训练小图就是人工画的区域框本身，
# 与推理侧的裁剪窗口（方案 margin=0 时同样等于区域框）完全一致
ENLARGE_RATIO = 0
# 标注框在窗口内至少可见的面积比例，达到才算该窗口内的钉
MIN_VISIBLE_RATIO = 0.5
# 参与标注的 labelme 标注类型
SHAPE_TYPE = "rectangle"
# 支持的图片扩展名
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']


def imread_unicode(file_path):
    """读取图片（兼容中文路径，cv2.imread 在 Windows 下读不了中文路径）"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(file_path, image):
    """写图片（兼容中文路径）"""
    ext = os.path.splitext(file_path)[1] or '.jpg'
    success, buffer = cv2.imencode(ext, image)
    if success:
        buffer.tofile(file_path)
    return success


def parse_labelme_boxes(shapes, image_name):
    """解析 labelme json 中的 rectangle 框，规整为左上、右下角坐标（可能出现反向拖框）"""
    boxes = []
    for shape in shapes:
        if shape.get('shape_type') != SHAPE_TYPE:
            continue
        (px1, py1), (px2, py2) = shape['points']
        x1, x2 = min(px1, px2), max(px1, px2)
        y1, y2 = min(py1, py2), max(py1, py2)
        if x2 <= x1 or y2 <= y1:
            print(f"警告：{image_name} 中有无效框 {shape['points']}，已跳过")
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def make_window(region_box, image_width, image_height):
    """区域框四周外扩 ENLARGE_RATIO 并裁到图像边界，返回整数 (x1,y1,x2,y2) 或 None（完全出界）"""
    x1, y1, x2, y2 = region_box
    box_width = x2 - x1
    box_height = y2 - y1
    crop_x1 = max(0, int(round(x1 - box_width * ENLARGE_RATIO)))
    crop_y1 = max(0, int(round(y1 - box_height * ENLARGE_RATIO)))
    crop_x2 = min(image_width, int(round(x2 + box_width * ENLARGE_RATIO)))
    crop_y2 = min(image_height, int(round(y2 + box_height * ENLARGE_RATIO)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return (crop_x1, crop_y1, crop_x2, crop_y2)


def find_stud_in_window(boxes, window):
    """找窗口内的钉：可见面积 ≥ MIN_VISIBLE_RATIO 的标注框。
    返回 (框或None, 候选数量)，多个候选时取可见面积最大的一个"""
    crop_x1, crop_y1, crop_x2, crop_y2 = window
    best_box = None
    best_visible_area = 0.0
    candidate_count = 0
    for x1, y1, x2, y2 in boxes:
        # 交集宽高 clamp 到 0：框在窗口外时差值为负，两个负数相乘会得正面积造成误判
        visible_width = max(0, min(x2, crop_x2) - max(x1, crop_x1))
        visible_height = max(0, min(y2, crop_y2) - max(y1, crop_y1))
        visible_area = visible_width * visible_height
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > 0 and visible_area >= box_area * MIN_VISIBLE_RATIO:
            candidate_count += 1
            if visible_area > best_visible_area:
                best_box = (x1, y1, x2, y2)
                best_visible_area = visible_area
    return best_box, candidate_count


def main():
    if len(sys.argv) < 3 or not os.path.isdir(sys.argv[1]):
        print("用法：python label_tool/step2_crop_to_yolo.py 图片文件夹路径 输出文件夹路径 [区域表方案号]")
        sys.exit(1)
    image_dir, output_dir = sys.argv[1], sys.argv[2]
    scheme = 0
    if len(sys.argv) > 3:
        try:
            scheme = int(sys.argv[3])
        except ValueError:
            print(f"区域表方案号 '{sys.argv[3]}' 不是整数")
            sys.exit(1)

    config = load_screw_regions(scheme=scheme)
    if config is None:
        print("错误：未找到有效的区域表方案，无法裁剪")
        sys.exit(1)
    regions = config['regions']
    single_region_scheme = len(regions) == 1
    print(f"区域表加载成功：{len(regions)} 个区域 | 外扩比例 {ENLARGE_RATIO}")

    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    image_files = sorted(
        name for name in os.listdir(image_dir)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    )
    if not image_files:
        print(f"文件夹中没有图片：{image_dir}")
        sys.exit(1)

    processed_images = 0     # 成功生成样本的原图数
    total_samples = 0        # 生成的小图总数（正样本 + 负样本）
    positive_samples = 0     # 有钉的正样本数
    skipped_no_json = 0      # 没有同名 json 的图（第一步没跑或已作废）
    skipped_read_fail = 0    # 图片读取失败
    skipped_outside_window = 0  # 单区域方案下标注框不在窗口内被跳过的样本数（异常）
    multi_stud_warnings = 0  # 窗口内出现多个钉的警告次数（区域表配置问题）
    out_of_image = 0         # 区域框完全在图像外无法裁剪的次数

    for name in image_files:
        stem = os.path.splitext(name)[0]
        json_path = os.path.join(image_dir, stem + '.json')
        if not os.path.exists(json_path):
            skipped_no_json += 1
            continue
        image = imread_unicode(os.path.join(image_dir, name))
        if image is None:
            print(f"图片读取失败，跳过：{name}")
            skipped_read_fail += 1
            continue
        image_height, image_width = image.shape[:2]

        with open(json_path, 'r', encoding='utf-8') as f:
            shapes = json.load(f).get('shapes', [])
        boxes = parse_labelme_boxes(shapes, name)

        # 每个区域固定生成一个样本（有钉 = 带标注 txt，无钉 = 空 txt 负样本）
        for region in regions:
            window = make_window(region['box'], image_width, image_height)
            if window is None:
                print(f"警告：{name} 的区域 {region['id']} 完全在图像外，无法裁剪")
                out_of_image += 1
                continue
            crop_x1, crop_y1, crop_x2, crop_y2 = window
            window_width = crop_x2 - crop_x1
            window_height = crop_y2 - crop_y1

            stud_box, candidate_count = find_stud_in_window(boxes, window)
            if candidate_count > 1:
                print(f"警告：{name} 的区域 {region['id']} 窗口内有 {candidate_count} 个钉，"
                      f"已取可见面积最大的一个。请检查区域表配置")
                multi_stud_warnings += 1
            # 单区域方案：json 里有框但都不在本窗口内 = 区域框/标注不一致的异常，
            # 警告并跳过该样本（防止有钉照片被静默当成负样本）；多区域方案下这是正常的空位负样本
            if stud_box is None and single_region_scheme and boxes:
                print(f"警告：{name} 的标注框不在区域 {region['id']} 窗口内，该样本已跳过（请检查区域框或标注）")
                skipped_outside_window += 1
                continue

            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_name = f"{stem}_r{region['id']}"
            if not imwrite_unicode(os.path.join(images_out, crop_name + '.jpg'), crop):
                print(f"小图写入失败：{crop_name}")
                continue

            with open(os.path.join(labels_out, crop_name + '.txt'), 'w', encoding='utf-8') as f:
                if stud_box is not None:
                    x1, y1, x2, y2 = stud_box
                    x_center = ((x1 + x2) / 2 - crop_x1) / window_width
                    y_center = ((y1 + y2) / 2 - crop_y1) / window_height
                    norm_width = (x2 - x1) / window_width
                    norm_height = (y2 - y1) / window_height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                    positive_samples += 1
            total_samples += 1

        processed_images += 1

    negative_samples = total_samples - positive_samples
    print(f"完成：处理 {processed_images} 张原图，生成 {total_samples} 个样本"
          f"（正样本 {positive_samples} + 负样本 {negative_samples}）")
    print(f"跳过：无 json {skipped_no_json} 张 | 读取失败 {skipped_read_fail} 张 | "
          f"标注框不在窗口 {skipped_outside_window} 张（异常） | 区域出图 {out_of_image} 次 | 多钉警告 {multi_stud_warnings} 次")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
