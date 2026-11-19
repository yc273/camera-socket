#!/usr/bin/env bash
# 从 GitHub Release 下载当前使用的模型权重到 model/
# 模型清单与 camera_socket-main/model_config.py 的 MODEL_PATHS 保持一致
set -e
cd "$(dirname "$0")"
REPO="yc273/camera-socket"
TAG="v2.0"
mkdir -p model
MODELS=(
  yolo11l-pose-stud-1280-16.pt
  yolo11x-pose-cross-1280-11.pt
  yolo11l-pose-hexstud-1280-6.pt
  slot-1280-2.pt
  yolo26x-detect-stud-256-3.pt
)
for m in "${MODELS[@]}"; do
  if [ -f "model/$m" ]; then
    echo "跳过（已存在）: model/$m"
    continue
  fi
  echo "下载: $m"
  curl -L -o "model/$m" "https://github.com/${REPO}/releases/download/${TAG}/${m}"
done
echo "完成。"
