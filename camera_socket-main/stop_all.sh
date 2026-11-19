#!/bin/bash
# 停止区域进程（默认全部；也可 ./stop_all.sh 1 3 只停部分区域）
AREAS=${*:-"1 2 3"}

for AREA in $AREAS; do
    # 匹配以 "--area N" 结尾的真实服务进程，避免误杀调用方 shell
    PIDS=$(pgrep -f "server_socket.py --area $AREA\$")
    if [ -n "$PIDS" ]; then
        echo "[stop_all] 停止区域 $AREA: $PIDS"
        kill $PIDS 2>/dev/null
    else
        echo "[stop_all] 区域 $AREA 未在运行"
    fi
done

# 等待退出，仍存活的强杀
sleep 3
for AREA in $AREAS; do
    PIDS=$(pgrep -f "server_socket.py --area $AREA\$")
    if [ -n "$PIDS" ]; then
        echo "[stop_all] 强制停止区域 $AREA: $PIDS"
        kill -9 $PIDS 2>/dev/null
    fi
done
echo "[stop_all] 完成"
