#!/usr/bin/env python3
"""TensorRT 模型编译脚本 —— 更换 .pt 模型后运行一次即可

用法（用 yolo 环境的 python 运行）:
    python3 export_engines.py                 # 检查全部模型，只编译缺失/过期的
    python3 export_engines.py --check         # 只查看缓存状态，不编译
    python3 export_engines.py --model cross   # 只处理指定模型（逗号分隔: stud,cross,u,slot,detect）

工作流程:
    1. 对每个模型的 .pt 计算内容指纹，检查对应的 .engine 缓存是否存在且匹配
    2. 缺失/过期 → 调 ultralytics 导出 TensorRT engine（首次编译较慢，x 型号可能 5~15 分钟）
    3. 若视觉服务(server_socket.py)正在运行，自动热替换已更新且开启中的模型（off→on），
       无需重启服务；服务没在跑则跳过，下次启动自动使用 engine

说明:
    - board 与 cross 共用同一权重，编译 cross 后两个模型都会热替换
    - engine 文件与 .pt 指纹绑定，随时可以删除，重新运行本脚本即可再生成
"""
import argparse
import json
import os
import socket
import sys
import time

from model_config import MODEL_PATHS, MODEL_TASKS
import engine_utils

HOST, PORT = "127.0.0.1", 65432
# board 与 cross 共用权重：cross 的 engine 变化时，两个模型都要热替换
SHARED = {"cross": ["cross", "board"]}


def send(cmd, timeout=180):
    """向视觉服务发一条指令；服务未运行返回 None"""
    try:
        with socket.create_connection((HOST, PORT), timeout=5) as s:
            s.settimeout(timeout)
            s.sendall((cmd + "\n").encode("utf-8"))
            data = b""
            while True:
                try:
                    chunk = s.recv(65536)
                except socket.timeout:
                    break
                if not chunk:
                    break
                data += chunk
                try:
                    return json.loads(data.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return None
    return None


def reload_via_server(names):
    """服务在跑时热替换指定模型（只动当前开启中的模型，不改变开关状态）"""
    status = send("model_status", timeout=10)
    if not status or not status.get("success"):
        print("\n[热替换] 视觉服务未运行，跳过（下次启动服务时自动使用 engine）")
        return
    models = status.get("models", {})
    for name in names:
        info = models.get(name)
        if not info:
            continue
        if not info.get("enabled"):
            print(f"[热替换] {name} 当前未启用，不动作（下次启用时自动用 engine）")
            continue
        print(f"[热替换] 重新加载 {name} ...", flush=True)
        send(f"model_{name}_off")
        r = send(f"model_{name}_on", timeout=300)
        ok = bool(r and r.get("success"))
        print(f"[热替换] {name} {'完成' if ok else '失败'}（之后可在服务日志看 [TensorRT]/[性能] 输出）", flush=True)


def main():
    parser = argparse.ArgumentParser(description="TensorRT engine 编译/缓存管理")
    parser.add_argument("--check", action="store_true", help="只查看状态，不编译")
    parser.add_argument("--model", type=str, default="", help="只处理指定模型，逗号分隔（board 会映射到 cross）")
    args = parser.parse_args()

    # board 共用 cross 权重，映射过去
    targets = list(MODEL_PATHS.keys())
    if args.model:
        targets = []
        for name in args.model.split(","):
            name = name.strip()
            targets.append("cross" if name == "board" else name)

    # 状态总览
    print("=" * 78)
    print(f"{'模型':<8}{'权重文件':<38}{'engine缓存':<12}{'状态'}")
    print("-" * 78)
    need_export = []
    for name in targets:
        pt = MODEL_PATHS.get(name)
        if not pt or not os.path.isfile(pt):
            print(f"{name:<8}{os.path.basename(pt) if pt else '-':<38}{'-':<12}权重文件不存在")
            continue
        cached = engine_utils.valid_engine(pt) is not None
        if cached:
            print(f"{name:<8}{os.path.basename(pt):<38}{'有':<12}缓存有效")
        else:
            size_mb = os.path.getsize(pt) / 1048576
            print(f"{name:<8}{os.path.basename(pt):<38}{'无':<12}待编译（{size_mb:.0f}MB）")
            need_export.append(name)
    print("=" * 78)

    if args.check:
        return

    if not need_export:
        print("\n所有模型缓存均有效，无需编译。")
        return

    reloaded = []
    for name in need_export:
        pt = MODEL_PATHS[name]
        try:
            engine_utils.export_engine(pt, task=MODEL_TASKS[name])
            reloaded.extend(SHARED.get(name, [name]))
        except KeyboardInterrupt:
            print("\n用户中断，已编译的缓存保留。")
            sys.exit(130)
        except Exception as e:
            print(f"[导出] {name} 编译失败: {e}", flush=True)
            print("       服务将回落使用 .pt 运行（不影响功能，只是没加速）。")

    if reloaded:
        time.sleep(1)
        reload_via_server(sorted(set(reloaded)))
    print("\n完成。engine 缓存位于 model/ 目录（*.engine，已被 git 忽略）。")


if __name__ == "__main__":
    main()
