#!/usr/bin/env bash
# 从 GitHub Release 下载全部模型权重到 model/
set -e
cd "$(dirname "$0")"
REPO="yc273/camera-socket"
TAG="v1.0"
mkdir -p model
MODELS=(
  detect-1280-8.pt
  slot-1280-2.pt
  stud-best-1280-15.pt
  u-l-1280-3.pt
  yolo11-cross-x-1280-8.pt
  yolo11-cross-x-1280-9.pt
  yolo11l-pose-hexstud-1280-4.pt
  yolo11l-pose-stud-1280-16.pt
  yolo11x-detect-1280-12.pt
  yolo11x-detect-1280-14.pt
  yolo11x-detect-stud-1280-16.pt
  yolo11x-pose-cross-1280-10.pt
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
