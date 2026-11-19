#!/bin/bash
# 一键拉起三个区域进程（按 1→2→3 顺序，等端口就绪再拉下一个，避免并发编译 engine）
cd "$(dirname "$0")"
mkdir -p logs

# 项目运行环境（默认 python3 无 cv2，需用 camera 环境）
PYTHON=${PYTHON:-/home/smartrobot/miniforge3/envs/camera/bin/python}

AREAS=${*:-"1 2 3"}  # 默认全部区域，也可 ./start_all.sh 1 2 只拉部分

for AREA in $AREAS; do
    # 从 areas.json 读端口
    PORT=$("$PYTHON" -c "import json;print(json.load(open('areas.json'))['$AREA']['port'])" 2>/dev/null)
    if [ -z "$PORT" ]; then
        echo "[start_all] 区域 $AREA 不存在或 areas.json 解析失败，跳过"
        continue
    fi

    # 已在运行则跳过（匹配以 "--area N" 结尾的真实服务进程）
    if pgrep -f "server_socket.py --area $AREA\$" > /dev/null; then
        echo "[start_all] 区域 $AREA 已在运行，跳过"
        continue
    fi

    echo "[start_all] 启动区域 $AREA (端口 $PORT)，日志 logs/area$AREA.log"
    nohup "$PYTHON" server_socket.py --area "$AREA" >> "logs/area$AREA.log" 2>&1 &

    # 等待端口监听就绪（最多 300s，首个进程可能需要编译/加载 engine）
    for i in $(seq 1 300); do
        if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
            echo "[start_all] 区域 $AREA 就绪"
            break
        fi
        sleep 1
    done
    sleep 2  # 额外错开，进一步降低 engine 缓存写冲突
done

echo "[start_all] 完成。运行中的区域进程："
pgrep -af "server_socket.py.*--area" || echo "  （无）"
