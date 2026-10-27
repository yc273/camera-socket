"""模型路径配置（唯一定义处）

realtime_test_origin.py（检测服务）与 export_engines.py（TensorRT 编译脚本）
都从这里取路径。新增/更换模型文件时，只需要改这个文件的 MODEL_PATHS。
"""
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_dir(name):
    """按优先级查找项目内的目录：脚本上一级 → 脚本同级 → 当前工作目录"""
    for candidate in (
        os.path.join(_SCRIPT_DIR, os.pardir, name),   # 项目根目录（当前布局）
        os.path.join(_SCRIPT_DIR, name),              # 脚本所在目录
        os.path.join(os.getcwd(), name),              # 运行时工作目录
    ):
        candidate = os.path.normpath(candidate)
        if os.path.isdir(candidate):
            return candidate
    return os.path.normpath(os.path.join(_SCRIPT_DIR, os.pardir, name))  # 兜底路径，缺失时由校验逻辑告警


# 模型统一放在 <项目根目录>/model/
MODEL_DIR = _find_dir("model")

# 逻辑名 → 权重路径。board 与 cross 共用同一权重，不单列。
MODEL_PATHS = {
    "stud":   os.path.join(MODEL_DIR, "yolo11l-pose-stud-1280-16.pt"),
    "cross":  os.path.join(MODEL_DIR, "yolo11x-pose-cross-1280-10.pt"),
    "u":      os.path.join(MODEL_DIR, "yolo11l-pose-hexstud-1280-4.pt"),
    "slot":   os.path.join(MODEL_DIR, "slot-1280-2.pt"),
    "detect": os.path.join(MODEL_DIR, "yolo26x-detect-stud-256-2.pt"),
}

# 逻辑名 → 任务类型。加载模型/引擎时必须显式指定，否则 ultralytics 按文件名猜测：
# slot 的文件名没有 pose/detect 关键字，会被误判成 detect，导致关键点解析失效。
MODEL_TASKS = {
    "stud": "pose",
    "cross": "pose",
    "board": "pose",
    "u": "pose",
    "slot": "pose",
    "detect": "detect",
}

# 各模型 TensorRT engine 导出尺寸，未列出的用 DEFAULT_EXPORT_IMGSZ。
# detect 是区域裁剪小图模型（stud_detect_regions.json 的 imgsz=256），
# engine 必须按推理同一尺寸导出，否则静态 shape 与 predict(imgsz=256) 不匹配会报错。
DEFAULT_EXPORT_IMGSZ = 1280
MODEL_EXPORT_IMGSZ = {
    "detect": 256,
}
