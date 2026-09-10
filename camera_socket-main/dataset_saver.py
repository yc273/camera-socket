#!/usr/bin/env python3
"""
数据集自动保存模块

收到客户端检测请求后，把当前帧（处理前 + 处理后）连同检测结果按类别
保存到 datasets/<类别>/ 下，为后续模型优化积累数据集。

目录结构：
    datasets/
      cross/
        cross_20260712-183025123_a1b2c3d4.jpg          # 处理后帧（喂给模型的）
        cross_20260712-183025123_a1b2c3d4_raw.jpg       # 原始帧
        cross_20260712-183025123_a1b2c3d4.json          # 标注（坐标/置信度/元信息）
      stud/
        ...
      u/  slot/  board/  frame/

文件命名： <类别>_<yyyyMMdd-HHmmssfff>_<uuid8>.<ext>

设计要点：
- 后台单线程写入（Queue + writer 线程），不阻塞 socket 响应
- 线程安全（多客户端并发请求时不会写冲突）
- 空帧/无数据时安全跳过
"""
import json
import os
import queue
import re
import threading
import time
import uuid
from datetime import datetime

import cv2
import numpy as np

# 根目录：camera_socket-main/datasets/
_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# 阶段照片根目录：camera_socket-main/stage_photos/（与 datasets 分开）
_STAGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage_photos")

# 请求 action -> 类别 映射
ACTION_CATEGORY = {
    "cross_points_without_granding": "cross",
    "screw_points": "stud",
    "u_points": "u",
    "slot_points": "slot",
    "board_angle": "board",
    "board_position": "board",
    "get_frame": "frame",
    "detect_count": "detect",
}


def category_of(action):
    """请求 action 转类别名，未知 action 归 other"""
    return ACTION_CATEGORY.get(action, "other")


class DatasetSaver:
    """按类别保存图片 + JSON 标注（后台线程写入）"""

    def __init__(self, base_dir=_BASE_DIR, enabled=True):
        self.base_dir = base_dir
        self.enabled = enabled
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._writer_thread = None
        self._running = False
        os.makedirs(self.base_dir, exist_ok=True)
        if self.enabled:
            self._start_writer()

    def _start_writer(self):
        self._running = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _writer_loop(self):
        """后台写入线程：从队列取任务，顺序写盘"""
        while self._running or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._write_task(task)
            except Exception as e:
                print(f"[DatasetSaver] 写入失败: {e}")
            finally:
                self._queue.task_done()

    def _write_task(self, task):
        """实际写盘：一张图片 + JSON"""
        if task.get("stage_dir"):
            self._write_stage_task(task)
            return
        category = task["category"]
        frame = task["frame"]
        result = task["result"]
        action = task["action"]
        ts = task["timestamp_str"]
        uid = task["uid"]

        cat_dir = os.path.join(self.base_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        stem = f"{category}_{ts}_{uid}"
        # 一张图：处理后帧（真正喂给模型的，gamma/contrast 处理过的）
        if frame is not None:
            img_path = os.path.join(cat_dir, f"{stem}.jpg")
            self._write_img(img_path, frame)

        # JSON 标注
        meta = {
            "category": category,
            "request_action": action,
            "timestamp": task["timestamp_iso"],
            "image_file": f"{stem}.jpg" if frame is not None else None,
            "image_shape": list(frame.shape) if frame is not None else None,
            "points": result,  # 该类别的检测结果（坐标/置信度等）
        }
        json_path = os.path.join(cat_dir, f"{stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=self._json_default)

    def _write_stage_task(self, task):
        """阶段照片写盘：stage_photos/<阶段>/<阶段>_<序号>_<时间戳>.jpg + JSON"""
        stage = task["stage"]
        seq = task["seq"]
        frame = task["frame"]

        stage_dir = os.path.join(_STAGE_BASE_DIR, stage)
        os.makedirs(stage_dir, exist_ok=True)

        stem = f"{stage}_{seq}_{task['timestamp_str']}"
        img_rel = os.path.join("stage_photos", stage, f"{stem}.jpg")
        if frame is not None:
            self._write_img(os.path.join(stage_dir, f"{stem}.jpg"), frame)

        meta = {
            "stage": stage,
            "seq": seq,
            "timestamp": task["timestamp_iso"],
            "image_file": f"{stem}.jpg" if frame is not None else None,
            "image_shape": list(frame.shape) if frame is not None else None,
        }
        with open(os.path.join(stage_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=self._json_default)

    @staticmethod
    def _sanitize(name):
        """清洗阶段名/序号，防止路径穿越（只保留字母数字下划线连字符）"""
        return re.sub(r"[^\w-]", "_", str(name)).strip("_")

    def save_stage(self, stage, seq, frame):
        """按阶段名+序号保存一帧到 stage_photos/<阶段>/（非阻塞）。

        文件名：<阶段>_<序号>_<yyyyMMdd-HHmmssfff>.jpg
        返回相对路径字符串；参数非法或无帧时返回 None。
        """
        if not self.enabled or frame is None:
            return None
        stage = self._sanitize(stage)
        seq = self._sanitize(seq)
        if not stage:
            return None
        now = datetime.now()
        task = {
            "stage_dir": True,
            "stage": stage,
            "seq": seq,
            "frame": frame.copy(),
            "timestamp_str": now.strftime("%Y%m%d-%H%M%S%f")[:-3],
            "timestamp_iso": now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        }
        self._queue.put(task)
        return os.path.join("stage_photos", stage,
                            f"{stage}_{seq}_{task['timestamp_str']}.jpg")

    @staticmethod
    def _write_img(path, frame):
        # 兼容单通道灰度图
        if frame.ndim == 2:
            cv2.imwrite(path, frame)
        else:
            cv2.imwrite(path, frame)

    @staticmethod
    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    def save(self, action, frame, result):
        """提交一个保存任务（非阻塞）。frame 为处理后帧（喂给模型那张）"""
        if not self.enabled:
            return
        if frame is None:
            return  # 无帧，跳过
        now = datetime.now()
        task = {
            "category": category_of(action),
            "action": action,
            "frame": frame.copy(),
            "result": result,
            "timestamp_str": now.strftime("%Y%m%d-%H%M%S%f")[:-3],
            "timestamp_iso": now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "uid": uuid.uuid4().hex[:8],
        }
        self._queue.put(task)

    def stop(self):
        """停止后台线程（通常不需要调用，守护线程随主进程退出）"""
        self._running = False
        if self._writer_thread:
            self._writer_thread.join(timeout=2.0)


# 全局单例（首次 import 时启动后台写入线程）
saver = DatasetSaver()
