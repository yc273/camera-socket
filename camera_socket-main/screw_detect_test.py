"""螺钉检测测试脚本：保留主路径（realtime_test_origin.py）ROI 区域裁剪检测的全部逻辑，
数据源从相机改为指定文件夹中的照片，用于离线验证新小模型和区域表的效果。

与主路径一致的逻辑：暗图 gamma 提亮、按 stud_detect_regions.json 区域表裁剪、
batch 推理、每区域取最高置信度1个框、conf_threshold 判定、坐标还原整图。
必须有有效的区域表，否则直接报错退出（不做任何兜底，避免误导性的测试结果）。

用法（在项目根目录、conda yolov8 环境运行）：
    python screw_detect_test.py 图片文件夹路径 [输出文件夹路径] [区域表方案号]
输出文件夹不传则默认 test_result/，方案号不传则默认 0。

输出图：绿框=有钉区域（检测框+区域id+置信度），红框=空钉区域（区域id），
左上角 Count: 检出数/总区域数。
"""
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from regions import load_screw_regions, crop_screw_regions
from model_config import MODEL_PATHS

# 待验证的小图模型（与主服务 MODEL_PATHS["detect"] 同一份配置）
MODEL_PATH = MODEL_PATHS["detect"]
DEFAULT_OUTPUT_DIR = "test_result"

# 与主路径一致的暗图提亮参数
DARK_THRESHOLD = 80
DARK_GAMMA = 2.2

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


def calculate_brightness(image):
    """计算图片平均亮度（0~255，数值越小越暗）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def adjust_gamma(image, gamma):
    """伽马提亮：gamma越大越亮"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def process_frame(frame):
    """与主路径一致：暗图 gamma 提亮，亮图不处理"""
    if calculate_brightness(frame) < DARK_THRESHOLD:
        return adjust_gamma(frame, DARK_GAMMA)
    return frame


def detect_by_regions(model, frame, config):
    """与主路径一致的区域裁剪检测：每区域取最高置信度1个框并还原整图坐标。
    返回 [(region, box或None, conf)，box 为整图坐标 [x1,y1,x2,y2]，None 表示判空钉]"""
    crops_with_info = crop_screw_regions(frame, config)
    region_results = []
    handled_ids = set()
    if crops_with_info:
        crops = [item[1] for item in crops_with_info]
        results_list = model.predict(
            source=crops,
            imgsz=config['imgsz'],
            conf=config['conf_threshold'],
            save=False,
            show=False,
            verbose=False
        )
        for (region, crop, offset_x, offset_y), result in zip(crops_with_info, results_list):
            handled_ids.add(region['id'])
            box = None
            conf = 0.0
            if result.boxes is not None and len(result.boxes) > 0:
                best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                conf = float(best_box.conf[0])
                box = [round(x1 + offset_x, 2), round(y1 + offset_y, 2),
                       round(x2 + offset_x, 2), round(y2 + offset_y, 2)]
            region_results.append((region, box, conf))
    # 裁剪被跳过的区域（如区域框在图像外）补判空钉
    for region in config['regions']:
        if region['id'] not in handled_ids:
            region_results.append((region, None, 0.0))
    return region_results


def draw_region_result(image, region_results):
    """画区域检测结果：绿框=有钉（检测框+id+conf），红框=空钉（区域框+id），返回标注图和检出数"""
    shown = image.copy()
    detected = 0
    for region, box, conf in region_results:
        x1, y1, x2, y2 = region['box']
        if box is None:
            cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(shown, str(region['id']), (x1, max(15, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            detected += 1
            bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(shown, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(shown, f"{region['id']}:{conf:.2f}", (bx1, max(15, by1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(shown, f"Count: {detected}/{len(region_results)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return shown, detected


def main():
    if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
        print("用法：python screw_detect_test.py 图片文件夹路径 [输出文件夹路径] [区域表方案号]")
        sys.exit(1)
    image_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    scheme = 0
    if len(sys.argv) > 3:
        try:
            scheme = int(sys.argv[3])
        except ValueError:
            print(f"区域表方案号 '{sys.argv[3]}' 不是整数")
            sys.exit(1)
    if os.path.abspath(output_dir) == os.path.abspath(image_dir):
        print("输出文件夹不能与图片文件夹相同（会覆盖原图）")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    config = load_screw_regions(scheme=scheme)
    if config is None:
        print("错误：未找到有效的区域表方案，无法进行区域裁剪检测")
        sys.exit(1)
    print(f"区域裁剪模式：{len(config['regions'])} 个区域 | imgsz={config['imgsz']} | "
          f"conf={config['conf_threshold']} | margin={config['margin']}")

    print(f"加载模型：{MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    image_files = sorted(
        name for name in os.listdir(image_dir)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    )
    if not image_files:
        print(f"文件夹中没有图片：{image_dir}")
        sys.exit(1)

    processed_count = 0
    failed_count = 0
    total_screws = 0

    for name in image_files:
        image = imread_unicode(os.path.join(image_dir, name))
        if image is None:
            print(f"{name}: 读取失败，跳过")
            failed_count += 1
            continue

        frame = process_frame(image)
        annotated, detected = draw_region_result(image, detect_by_regions(model, frame, config))

        if not imwrite_unicode(os.path.join(output_dir, name), annotated):
            print(f"{name}: 结果图写入失败")
            failed_count += 1
            continue
        total_screws += detected
        processed_count += 1
        print(f"{name}: 检出 {detected} 颗")

    print(f"\n汇总：处理 {processed_count} 张 | 读取/写入失败 {failed_count} 张 | "
          f"共检出 {total_screws} 颗螺钉 | 平均 {total_screws / processed_count:.1f} 颗/张"
          if processed_count else "\n没有成功处理的图片")
    print(f"结果图已保存至：{output_dir}")


if __name__ == "__main__":
    main()
