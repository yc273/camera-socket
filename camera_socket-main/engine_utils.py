"""TensorRT engine 缓存管理

engine 文件与 .pt 一一对应，文件名带 .pt 的内容指纹（md5 前 10 位）：
    <权重名>.<指纹>.engine   例: yolo11x-pose-cross-1280-10.a1b2c3d4e5.engine

.pt 换新后指纹变化 → 旧 engine 文件名对不上 → 自动视为失效，重新导出即可。
检测程序加载时调用 valid_engine() 优先使用匹配的 engine，没有则回落 .pt。
"""
import hashlib
import os
import time

# 导出参数：与实时检测的 predict(imgsz=1280) 保持一致；half=True 用 FP16（Jetson Tensor Core 硬件加速）
EXPORT_IMGSZ = 1280
EXPORT_HALF = True


def file_fingerprint(path, chunk_mb=4):
    """计算文件内容指纹（md5 前 10 位 hex），换模型后指纹必然变化"""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_mb * 1024 * 1024)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()[:10]


def engine_path_for(pt_path):
    """与该 .pt 内容指纹绑定的 engine 缓存路径（与 .pt 同目录）"""
    stem = os.path.splitext(os.path.basename(pt_path))[0]
    fp = file_fingerprint(pt_path)
    return os.path.join(os.path.dirname(pt_path), f"{stem}.{fp}.engine")


def valid_engine(pt_path):
    """返回与当前 .pt 匹配的 engine 缓存路径；不存在则返回 None"""
    if not pt_path or not os.path.isfile(pt_path):
        return None
    engine = engine_path_for(pt_path)
    return engine if os.path.isfile(engine) else None


def export_engine(pt_path, task, device=0):
    """把 .pt 编译为 TensorRT engine（首次编译较慢，x 型号可能 5~15 分钟），返回 engine 路径"""
    from ultralytics import YOLO  # 延迟导入：脚本启动时不必拉起 torch

    print(f"[导出] 开始编译 TensorRT engine: {pt_path} (task={task})", flush=True)
    print(f"[导出] imgsz={EXPORT_IMGSZ} half={EXPORT_HALF}（首次编译耗时较长，请耐心等待）", flush=True)
    t0 = time.time()
    model = YOLO(pt_path, task=task)
    out = model.export(
        format="engine",
        imgsz=EXPORT_IMGSZ,
        half=EXPORT_HALF,
        device=device,
        batch=1,
        dynamic=False,
        workspace=4,
    )
    target = engine_path_for(pt_path)
    # ultralytics 产物固定叫 <stem>.engine，重命名为带指纹的缓存名
    if os.path.abspath(out) != os.path.abspath(target):
        os.replace(out, target)
    _cleanup_stale_engines(pt_path, keep=target)
    print(f"[导出] 完成: {target} (耗时 {time.time() - t0:.0f}s)", flush=True)
    return target


def _cleanup_stale_engines(pt_path, keep):
    """删除同一权重的其他（过期）engine 缓存，避免 model/ 目录堆积"""
    stem = os.path.splitext(os.path.basename(pt_path))[0]
    model_dir = os.path.dirname(pt_path)
    for name in os.listdir(model_dir):
        if name.startswith(stem + ".") and name.endswith(".engine"):
            path = os.path.join(model_dir, name)
            if os.path.abspath(path) != os.path.abspath(keep):
                os.remove(path)
                print(f"[清理] 移除过期引擎缓存: {name}", flush=True)
