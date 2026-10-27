"""螺钉检测区域表：读取人工标定的"可能放螺钉"的矩形区域配置（stud_detect_regions.json），
螺钉检测时按区域裁剪小图逐个识别，替换整图识别，降低错检漏检。"""
import json
import os

# 区域表固定在本文件所在的项目根目录（不受运行时工作目录影响）
REGIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stud_detect_regions.json")


def load_screw_regions(scheme=0, path=REGIONS_FILE):
    """读取并校验指定序号的裁剪方案（json 的 schemes 字典按序号选择）。
    只支持新格式（顶层必须有 schemes 键）。任何加载失败都返回 None"""
    if not os.path.exists(path):
        print(f"警告：未找到区域表 {path}，螺钉检测将使用整图识别")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if 'schemes' not in config:
            print(f"错误：区域表 {path} 不是新格式（缺少 schemes 键），无法加载")
            return None
        key = str(scheme)
        if key not in config['schemes']:
            print(f"错误：区域表不存在方案 {scheme}，可用方案：{sorted(config['schemes'].keys(), key=int)}")
            return None
        scheme_config = config['schemes'][key]

        regions = []
        for i, item in enumerate(scheme_config.get('regions', [])):
            x1, y1, x2, y2 = item['box']
            if x2 <= x1 or y2 <= y1:
                print(f"警告：区域 {item.get('id', i + 1)} 的 box {item['box']} 无效（需 x2>x1、y2>y1），已跳过")
                continue
            regions.append({'id': item.get('id', i + 1), 'box': [x1, y1, x2, y2]})
        if not regions:
            print(f"警告：区域表方案 {scheme} 中没有有效区域，螺钉检测将使用整图识别")
            return None
        print(f"螺钉检测区域表加载成功：方案 {scheme}，共 {len(regions)} 个区域")
        return {
            'scheme': scheme,
            'margin': scheme_config.get('margin', 0),
            'conf_threshold': scheme_config.get('conf_threshold', 0.5),
            'imgsz': scheme_config.get('imgsz', 320),
            'regions': regions
        }
    except Exception as e:
        print(f"警告：区域表 {path} 加载失败（{e}），螺钉检测将使用整图识别")
        return None


def crop_screw_regions(frame, config):
    """按区域表裁剪小图，返回 [(区域, 小图, 裁剪起点x, 裁剪起点y), ...]

    裁剪时向外扩 margin 再裁到图像边界内，防止标注矩形太紧切到螺钉边缘；
    偏移量用于把小图上检测到的框坐标还原回整图坐标。
    """
    frame_height, frame_width = frame.shape[:2]
    margin = config['margin']
    crops = []
    for region in config['regions']:
        x1, y1, x2, y2 = region['box']
        crop_x1 = max(0, x1 - margin)
        crop_y1 = max(0, y1 - margin)
        crop_x2 = min(frame_width, x2 + margin)
        crop_y2 = min(frame_height, y2 + margin)
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue
        crops.append((region, crop, crop_x1, crop_y1))
    return crops
