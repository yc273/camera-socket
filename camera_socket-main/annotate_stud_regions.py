"""螺钉检测区域标注工具：
框选物料台上所有可能放螺钉的矩形区域，生成 stud_detect_regions.json 供螺钉检测使用。
支持多版本裁剪方案（schemes 字典按序号选择），旧格式文件编辑保存时自动升级为新格式。

用法（在项目根目录、conda yolov8 环境下运行）：
    python annotate_stud_regions.py                        标注 DEFAULT_IMAGE，编辑方案 0
    python annotate_stud_regions.py 某张图片.jpg            标注命令行指定的图片
    python annotate_stud_regions.py 某张图片.jpg 1          编辑/新建方案 1
指定不存在的方案号 = 新建该方案；旧格式文件仅含方案 0。

操作：
    鼠标左键拖拽    画一个区域（松开左键确定）
    u              撤销最近画的区域
    s 或 回车       保存 JSON 并退出
    q 或 ESC       不保存退出

说明：图像按比例缩小显示（大图适配屏幕），保存的坐标是原图像素坐标；
保存时其他方案的配置原样保留，只更新当前编辑的方案。
"""
import sys
import json
import os
import cv2

# 默认标注图片（命令行传入图片路径时可覆盖），可先用 socket 服务 get_frame 保存一帧
DEFAULT_IMAGE = "frame.jpg"
# 生成的区域表路径（realtime_test_origin.py 读取的就是这个文件；锚定本脚本目录，
# 与 regions.py 的 REGIONS_FILE 同一位置，不受运行时工作目录影响）
OUTPUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stud_detect_regions.json")
# 显示窗口最大宽度，超过则按比例缩小显示
MAX_DISPLAY_WIDTH = 1280
# 新建方案时写入的默认参数（之后可手动修改 JSON 调整）；margin=0 = 推理裁剪窗口就是人工框本身
DEFAULT_MARGIN = 0
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_IMGSZ = 256

# 正在拖拽的起点和当前点（显示坐标系），None 表示没有在拖拽
drawing_start = None
drawing_current = None
# 编辑状态：当前方案的区域列表（原图坐标系）、方案号、方案参数、文件中的全部方案
regions = []
current_scheme = 0
current_params = {}
all_schemes = None


def mouse_callback(event, x, y, flags, param):
    """鼠标事件：左键拖拽画区域（显示坐标，保存时换算回原图坐标）"""
    global drawing_start, drawing_current
    scale = param['scale']
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_start = (x, y)
        drawing_current = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing_start is not None:
        drawing_current = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and drawing_start is not None:
        # 显示坐标 → 原图坐标，并规整为左上角、右下角
        x1 = round(min(drawing_start[0], x) / scale)
        y1 = round(min(drawing_start[1], y) / scale)
        x2 = round(max(drawing_start[0], x) / scale)
        y2 = round(max(drawing_start[1], y) / scale)
        drawing_start = None
        drawing_current = None
        if x2 > x1 and y2 > y1:
            regions.append({'id': len(regions) + 1, 'box': [x1, y1, x2, y2]})
            print(f"已添加区域 {len(regions)}: [{x1}, {y1}, {x2}, {y2}]（共 {len(regions)} 个）")


def draw_all_regions(display_image, scale):
    """把所有已标注区域和正在拖拽的矩形画到显示图上"""
    shown = display_image.copy()
    for region in regions:
        x1, y1, x2, y2 = region['box']
        p1 = (int(x1 * scale), int(y1 * scale))
        p2 = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(shown, p1, p2, (0, 255, 0), 2)
        cv2.putText(shown, str(region['id']), (p1[0], max(15, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if drawing_start is not None and drawing_current is not None:
        cv2.rectangle(shown, drawing_start, drawing_current, (0, 255, 255), 1)
    return shown


def load_existing(scheme):
    """读取已有区域表中指定方案的内容，填充全局编辑状态。
    只支持新格式（顶层必须有 schemes 键）；文件不存在或损坏则从空白开始"""
    global regions, current_scheme, current_params, all_schemes
    regions = []
    current_scheme = scheme
    current_params = {'margin': DEFAULT_MARGIN, 'conf_threshold': DEFAULT_CONF_THRESHOLD,
                      'imgsz': DEFAULT_IMGSZ}
    all_schemes = None
    if not os.path.exists(OUTPUT_JSON):
        print(f"区域表 {OUTPUT_JSON} 不存在，将新建方案 {scheme}")
        return
    try:
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取已有区域表失败（{e}），从空白开始标注，保存时将覆盖原文件")
        return
    if 'schemes' not in data:
        print("错误：区域表不是新格式（缺少 schemes 键），请先手动转换格式，本次从空白开始且保存将覆盖原文件")
        return
    all_schemes = data['schemes']
    key = str(scheme)
    if key in all_schemes:
        scheme_data = all_schemes[key]
    else:
        print(f"方案 {scheme} 不存在，将新建（已有方案：{sorted(all_schemes.keys(), key=int)}）")
        return
    regions = scheme_data.get('regions', [])
    current_params = {'margin': scheme_data.get('margin', DEFAULT_MARGIN),
                      'conf_threshold': scheme_data.get('conf_threshold', DEFAULT_CONF_THRESHOLD),
                      'imgsz': scheme_data.get('imgsz', DEFAULT_IMGSZ)}
    print(f"已加载方案 {current_scheme} 的 {len(regions)} 个已有区域，可继续追加或按 u 撤销")


def save_regions():
    """保存区域表（统一为新格式：schemes 字典），其他方案原样保留，只更新当前编辑的方案"""
    global all_schemes
    scheme_data = {
        'margin': current_params['margin'],
        'conf_threshold': current_params['conf_threshold'],
        'imgsz': current_params['imgsz'],
        'regions': regions
    }
    if all_schemes is None:
        all_schemes = {}
    all_schemes[str(current_scheme)] = scheme_data
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'schemes': all_schemes}, f, ensure_ascii=False, indent=2)
    print(f"已保存方案 {current_scheme} 的 {len(regions)} 个区域到 {OUTPUT_JSON}"
          f"（共 {len(all_schemes)} 个方案：{sorted(all_schemes.keys(), key=int)}）")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    scheme = 0
    if len(sys.argv) > 2:
        try:
            scheme = int(sys.argv[2])
        except ValueError:
            print(f"方案号 '{sys.argv[2]}' 不是整数")
            sys.exit(1)
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片：{image_path}")
        print("可以先运行 socket 服务后发送 get_frame 保存一帧，或命令行传入图片路径：")
        print("    python annotate_stud_regions.py 某张图片.jpg")
        sys.exit(1)

    load_existing(scheme)

    # 大图缩小显示，所有标注坐标按比例换算回原图
    height, width = image.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / width)
    if scale < 1.0:
        display_image = cv2.resize(image, (int(width * scale), int(height * scale)))
    else:
        display_image = image.copy()
    print(f"标注图片：{image_path}（{width}x{height}，显示缩放 {scale:.2f}，保存坐标为原图像素坐标）")
    print(f"当前编辑方案：{current_scheme}")
    print("操作：左键拖拽画区域 | u 撤销 | s/回车 保存退出 | q/ESC 不保存退出")

    cv2.namedWindow("annotate", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("annotate", mouse_callback, {'scale': scale})

    while True:
        cv2.imshow("annotate", draw_all_regions(display_image, scale))
        key = cv2.waitKey(20) & 0xFF
        if key == ord('u'):
            if regions:
                removed = regions.pop()
                print(f"已撤销区域 {removed['id']}: {removed['box']}（剩 {len(regions)} 个）")
            else:
                print("没有可撤销的区域")
        elif key in (ord('s'), 13):
            save_regions()
            break
        elif key in (ord('q'), 27):
            print("未保存退出")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
