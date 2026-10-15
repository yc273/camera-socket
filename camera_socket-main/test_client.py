#!/usr/bin/env python3
"""
Socket 客户端测试脚本

连接 server_socket.py 服务端，发送检测/控制指令，接收并解析响应。
用于端到端测试 socket 服务是否正常。

用法（确保 server_socket.py 已启动）:
    # 交互模式：逐条输入指令
    python test_client.py

    # 指定 server 地址
    python test_client.py --host 192.168.3.198 --port 65432

    # 直接执行一条指令并退出
    python test_client.py --cmd cross_points_without_granding

    # 交互式菜单（推荐）
    python test_client.py --menu

注意：无相机连接时，检测类指令会返回 "相机未连接" 提示帧（属正常），
      控制类指令（model_status / model_*_on/off）不受影响，可正常测试。
"""
import argparse
import base64
import json
import socket
import sys
import time

# 所有可用指令
COMMANDS = {
    # 检测类（无相机时返回提示帧）
    "1": ("cross_points_without_granding", "十字交叉点坐标"),
    "2": ("screw_points", "螺钉坐标"),
    "3": ("board_angle", "装板角度"),
    "4": ("board_position", "装板位置"),
    "5": ("u_points", "U型件坐标"),
    "6": ("slot_points", "十字接缝坐标"),
    "7": ("detect_count", "检测目标数量"),
    "8": ("get_frame", "获取当前帧（base64图像）"),
    # 控制类（不受相机状态影响）
    "9": ("model_status", "查询所有模型状态"),
    "10": ("model_cross_on", "启用 cross 模型"),
    "11": ("model_cross_off", "禁用 cross 模型"),
    "12": ("model_stud_on", "启用 stud 模型"),
    "13": ("model_detect_on", "启用 detect 模型"),
    # 拍照类
    "14": ("take_photo", "拍照：保存当前帧到 data/manual/"),
    # 基础
    "0": ("ping", "心跳/连通测试"),
}


def send_command(sock, cmd, timeout=5.0):
    """发送一条指令并接收响应"""
    sock.settimeout(timeout)
    t0 = time.perf_counter()
    try:
        sock.sendall((cmd + "\n").encode("utf-8"))
    except Exception as e:
        return None, f"发送失败: {e}", 0

    # 接收响应（裸 JSON，读到连接关闭或超时；get_frame 响应较大）
    data = b""
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            # 尝试解析，成功则提前结束（避免等超时）
            try:
                json.loads(data.decode("utf-8"))
                break
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    except socket.timeout:
        if not data:
            return None, "接收超时（无响应）", time.perf_counter() - t0
    except Exception as e:
        return None, f"接收失败: {e}", time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    try:
        return json.loads(data.decode("utf-8")), None, elapsed
    except Exception as e:
        return None, f"JSON 解析失败: {e}（原始: {data[:200]!r}）", elapsed


def print_response(resp, err, elapsed):
    """格式化打印响应"""
    print(f"  耗时: {elapsed * 1000:.1f} ms")
    if err:
        print(f"  ❌ {err}")
        return
    if resp is None:
        print("  ❌ 无响应")
        return

    # 相机未连接提示
    if resp.get("camera_status") == "disconnected":
        print(f"  ⚠️  {resp.get('message', '相机未连接')}")
        print(f"     （server 正常，只是当前无相机 → 数据集不保存，符合预期）")
        return

    # 帧响应（base64 太长，只显示摘要）
    if "frame" in resp:
        frame_size = len(resp["frame"])
        print(f"  ✅ 收到图像帧: {frame_size} bytes (base64), shape={resp.get('shape')}")
        print(f"     success={resp.get('success')}")
        return

    # 模型状态（展开打印）
    if "models" in resp:
        print("  ✅ 模型状态:")
        for name, info in resp.get("models", {}).items():
            print(f"     {name:8s}: loaded={info.get('loaded')}, enabled={info.get('enabled')}")
        return

    # enable/disable 结果
    if resp.get("success") and "message" in resp and len(resp) <= 3:
        print(f"  ✅ {resp.get('message')}")
        return

    # 检测结果（坐标等）
    print(f"  ✅ 响应:")
    for k, v in resp.items():
        if k == "timestamp":
            continue
        s = json.dumps(v, ensure_ascii=False)
        if len(s) > 200:
            s = s[:200] + "...(截断)"
        print(f"     {k}: {s}")
    if "timestamp" in resp:
        ts = resp["timestamp"]
        print(f"     timestamp: 处理 {ts.get('processing_time_ms', '?')}ms, 往返 {ts.get('time_diff_ms', '?')}ms")


def run_menu(host, port):
    """菜单模式"""
    print(f"\n连接到 {host}:{port} ...")
    try:
        sock = socket.create_connection((host, port), timeout=5)
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print(f"   确认 server_socket.py 已启动，且 {host}:{port} 可达")
        return

    print(f"✅ 已连接 {host}:{port}\n")

    try:
        while True:
            print("=" * 50)
            print("可用指令（输入编号或指令名，q 退出）:")
            for k, (cmd, desc) in COMMANDS.items():
                print(f"  {k:>2s}. {cmd:32s} {desc}")
            print("=" * 50)
            choice = input(">>> ").strip()

            if choice.lower() in ("q", "quit", "exit"):
                print("退出。")
                break
            if not choice:
                continue

            # 编号 → 指令名
            if choice in COMMANDS:
                cmd = COMMANDS[choice][0]
            else:
                cmd = choice  # 直接输入指令名

            print(f"\n[发送] {cmd}")
            resp, err, elapsed = send_command(sock, cmd)
            print_response(resp, err, elapsed)

            # 单连接复用：server 端 handle_client 是 while True 循环，支持多请求
            print()
    except KeyboardInterrupt:
        print("\n中断，退出。")
    finally:
        sock.close()


def run_single(host, port, cmd):
    """单指令模式"""
    print(f"连接到 {host}:{port} ...")
    try:
        sock = socket.create_connection((host, port), timeout=5)
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return 1

    print(f"[发送] {cmd}")
    resp, err, elapsed = send_command(sock, cmd)
    print_response(resp, err, elapsed)
    sock.close()
    return 0 if err is None else 1


def main():
    p = argparse.ArgumentParser(description="server_socket 客户端测试")
    p.add_argument("--host", default="192.168.3.198", help="server IP（默认 192.168.3.198）")
    p.add_argument("--port", type=int, default=65432, help="server 端口（默认 65432）")
    p.add_argument("--cmd", default=None, help="直接执行一条指令并退出")
    p.add_argument("--menu", action="store_true", help="菜单交互模式")
    args = p.parse_args()

    if args.cmd:
        sys.exit(run_single(args.host, args.port, args.cmd))
    else:
        # 默认进菜单
        run_menu(args.host, args.port)


if __name__ == "__main__":
    main()
