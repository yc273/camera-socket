# camera-socket

基于 Jetson + 海康工业相机 + YOLO11 的实时位姿检测服务（光伏支架安装板/交叉梁/六角螺柱/插槽/螺钉检测）。

## 目录结构

```
camera_socket-main/   # 主代码（相机控制、实时检测、socket 服务）
  realtime_test_origin.py   # 实时位姿检测器（核心）
  server_socket.py          # TCP socket 服务端
  camera.py                 # 海康相机封装
model/                # 模型权重（不入库，见下方"模型下载"）
data/                 # 运行时数据
```

## 环境要求

- Python 3.10+，依赖见 `camera_socket-main/requirements.txt`
- 海康 MVS SDK（Linux 下需安装并设置 `MVCAM_COMMON_RUNENV` 环境变量，运行时加载其 `libMvCameraControl.so`；Windows DLL 不随仓库分发）
- GPU 推理依赖 `ultralytics` + TensorRT/PyTorch（Jetson 平台）

## 模型下载

模型权重单文件超过 GitHub 100MB 限制，通过 [Releases](https://github.com/yc273/camera-socket/releases) 分发。克隆后执行：

```bash
./download_models.sh   # 自动下载全部 .pt 权重到 model/
```

## 快速开始

```bash
pip install -r camera_socket-main/requirements.txt
./download_models.sh
python camera_socket-main/realtime_test_origin.py   # 单独运行检测
python camera_socket-main/server_socket.py          # 以 socket 服务运行
```

模型路径已改为按脚本位置自动定位（优先项目根 `model/`），换机器无需改代码。
