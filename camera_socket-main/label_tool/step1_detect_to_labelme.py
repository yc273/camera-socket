"""第一步：用区域表裁剪 + 小模型逐区域检测辅助生成标注文件（替代老模型整图检测）。
检测管线与线上推理一致：暗图 gamma 提亮 → 按 stud_detect_regions.json 区域裁剪小图
（margin 外扩）→ batch 推理 → 每区域取最高置信度 1 个框 → 坐标还原整图 → 生成 labelme json。
人工在 labelme 中审核调整（删除错检框、补画漏检框）后，用 step2_crop_to_yolo.py 生成训练集。

用法（在项目根目录、conda yolov8 环境运行）：
    python label_tool/step1_detect_to_labelme.py 图片文件夹路径 [区域表方案号]
方案号不传则默认 0。

注意：已存在同名 json 的图片会跳过，防止覆盖人工审核过的结果。
"""
import os
import sys
import json
import cv2
import numpy as np
from ultralytics import YOLO

# 把项目根目录加入模块搜索路径（本脚本在 label_tool 子目录中，regions.py 在项目根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regions import load_screw_regions, crop_screw_regions

# 辅助标注用的小图模型（与 screw_detect_test.py 一致）
MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "model", "yolo11x-detect-stud-256-1.pt"))
# 辅助标注的置信度阈值（宁多勿漏，错检人工删、漏检人工补）
CONF = 0.25
# 与主路径一致的暗图提亮参数
DARK_THRESHOLD = 80
DARK_GAMMA = 2.2
# labelme 里的类别名（txt 的 class_id 固定为 0，与这个名字无关）
LABEL_NAME = "stud"
# 支持的图片扩展名
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']


def imread_unicode(file_path):
    """读取图片（兼容中文路径，cv2.imread 在 Windows 下读不了中文路径）"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def process_frame(frame):
    """与主路径一致：暗图 gamma 提亮，亮图不处理"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < DARK_THRESHOLD:
        inv_gamma = 1.0 / DARK_GAMMA
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame, table)
    return frame


def detect_by_regions(model, frame, config):
    """与线上推理一致的区域裁剪检测：每区域取最高置信度 1 个框并还原整图坐标。
    返回整图坐标框列表 [x1, y1, x2, y2]"""
    crops_with_info = crop_screw_regions(frame, config)
    if not crops_with_info:
        return []
    crops = [item[1] for item in crops_with_info]
    results_list = model.predict(
        source=crops,
        imgsz=config['imgsz'],
        conf=CONF,
        save=False,
        show=False,
        verbose=False
    )
    boxes = []
    for (region, crop, offset_x, offset_y), result in zip(crops_with_info, results_list):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        boxes.append([round(x1 + offset_x, 2), round(y1 + offset_y, 2),
                      round(x2 + offset_x, 2), round(y2 + offset_y, 2)])
    return boxes


def build_labelme_json(image_name, boxes, image_height, image_width):
    """构造 labelme 格式的 json 内容（imageData 不嵌入图片数据，labelme 打开时从同目录加载图片）"""
    shapes = []
    for x1, y1, x2, y2 in boxes:
        shapes.append({
            'label': LABEL_NAME,
            'points': [[float(x1), float(y1)], [float(x2), float(y2)]],
            'group_id': None,
            'description': '',
            'shape_type': 'rectangle',
            'flags': {}
        })
    return {
        'version': '5.2.1',
        'flags': {},
        'shapes': shapes,
        'imagePath': image_name,
        'imageData': None,
        'imageHeight': int(image_height),
        'imageWidth': int(image_width)
    }


def main():
    if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
        print("用法：python label_tool/step1_detect_to_labelme.py 图片文件夹路径 [区域表方案号]")
        sys.exit(1)
    image_dir = sys.argv[1]
    scheme = 0
    if len(sys.argv) > 2:
        try:
            scheme = int(sys.argv[2])
        except ValueError:
            print(f"区域表方案号 '{sys.argv[2]}' 不是整数")
            sys.exit(1)

    config = load_screw_regions(scheme=scheme)
    if config is None:
        print("错误：未找到有效的区域表方案，无法辅助生成标注")
        sys.exit(1)
    print(f"区域裁剪辅助标注：方案 {scheme}，{len(config['regions'])} 个区域 | "
          f"imgsz={config['imgsz']} | margin={config['margin']} | conf={CONF}")

    image_files = sorted(
        name for name in os.listdir(image_dir)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    )
    if not image_files:
        print(f"文件夹中没有图片：{image_dir}")
        sys.exit(1)

    # 已有同名 json 的跳过，防止覆盖人工审核结果
    todo_files = []
    skipped_count = 0
    for name in image_files:
        json_path = os.path.join(image_dir, os.path.splitext(name)[0] + '.json')
        if os.path.exists(json_path):
            skipped_count += 1
        else:
            todo_files.append(name)
    print(f"共 {len(image_files)} 张图片，跳过已有 json 的 {skipped_count} 张，待处理 {len(todo_files)} 张")
    if not todo_files:
        print("全部已有 json，无需处理")
        return

    print(f"加载模型：{MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    total_boxes = 0
    processed_count = 0
    for name in todo_files:
        image = imread_unicode(os.path.join(image_dir, name))
        if image is None:
            print(f"读取失败，跳过：{name}")
            continue
        image_height, image_width = image.shape[:2]

        frame = process_frame(image)
        boxes = detect_by_regions(model, frame, config)

        labelme_json = build_labelme_json(name, boxes, image_height, image_width)
        json_path = os.path.join(image_dir, os.path.splitext(name)[0] + '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_json, f, ensure_ascii=False, indent=2)
        total_boxes += len(boxes)
        processed_count += 1
        print(f"{name}: 检出 {len(boxes)} 个框")

    print(f"完成：处理 {processed_count} 张，共生成 {total_boxes} 个标注框，json 已保存在 {image_dir}")
    print("请用 labelme 打开该文件夹审核：删除错检框、补画漏检框，再运行 step2_crop_to_yolo.py")


if __name__ == "__main__":
    main()
