# 视觉相机识别系统（放料台螺钉检测）

基于 Jetson + 海康工业相机 + YOLO（ultralytics，TensorRT 加速）的实时视觉检测系统，通过 Socket 服务向客户端提供检测结果。

本系统核心业务为**放料台螺钉检测**，检测方案为**单钉单拍 + 单区域裁剪**（见第一节）；同时保留早期功能：螺钉关键点、装板十字线（角度/位置）、U 型件六角螺套、十字接缝。

## 一、放料台螺钉检测逻辑的演进

### 第一代：一张图同时检测约 40 个位置

- 做法：人工标定放料台上约 40 个可能放螺钉的矩形区域，检测时逐区域裁剪小图，模型判断每个位置有无螺钉并统计总数。
- 问题（现场实测）：
  - 同一颗螺钉放在不同位置时，**相机拍到的姿态/角度各不相同**，模型需要覆盖大量形态变化；叠加**光照不均**影响，错检漏检难以压下去。
  - 以"一张图 40 个位置全对"为判定标准，**整图错误率最高约 10%**——折算到单颗螺钉的错率约 0.26%，且错误集中在少数极端姿态的位置上。工业场景不可接受。
  - 训练样本量大、迭代成本高。

### 第二代（当前）：单钉单拍，单区域裁剪

- 前提：机构/上位机保证**一次只检测一颗螺钉**，螺钉每次出现在画面中基本相同的区域、姿态基本一致。
- 做法：人工只标定 **1 个矩形区域**（它就是裁剪窗口），检测时裁剪这一小块小图（`yolo26x-detect-stud-256-2.pt`，imgsz=256），模型判断这一颗螺钉的有无（输出 0/1）。
- 关键设计：
  - **人工框 = 裁剪窗口**：训练小图的裁剪（step2 工具）与线上推理的裁剪（服务）使用同一个人工框，没有任何外扩和偏移，训练分布与推理分布完全一致（这是踩过"训练/推理裁剪窗口不一致导致大量漏检"的坑之后定稿的设计）。
  - **区域表加载失败时 detect 直接不检测**，不做整图兜底（256 小图模型跑整图必然大量误检漏检）。
  - 目标域收窄后，很小的模型 + 少量样本即可达到很高准确率；错误只影响单颗螺钉，可疑帧可立即重拍。
- 使用前提与注意：
  - **机械一致性**是收益成立的前提：螺钉必须稳定落在标定区域内。标定后用 `label_tool/check_region_fit.py` 批量目检确认（见第五节）。
  - 相机或机械位置变动后必须重新标定区域。
  - 框外区域模型完全看不到——画框时把螺钉的波动范围和必要的背景一起框进去。
- 接口影响：`detect_count` 的语义为 **0/1**（本次检测的这一颗螺钉有无）。

## 二、环境要求

- Jetson（Linux）+ Python 3.10+，依赖见 `requirements.txt`（ultralytics + TensorRT/PyTorch）
- 海康威视工业相机，Linux 下需安装 MVS SDK 并设置 `MVCAM_COMMON_RUNENV`（运行时加载 `libMvCameraControl.so`）
- 模型权重在仓库根 `model/` 目录（路径由 `model_config.py` 自动定位，换机器无需改代码）。**detect 模型（生产）**：

| 用途 | 文件 | 推理方式 |
|---|---|---|
| **单钉螺钉检测（生产）** | `model/yolo26x-detect-stud-256-2.pt` | 区域裁剪小图，imgsz=256 |
| 螺钉关键点（旧功能） | `model/yolo11l-pose-stud-1280-16.pt` | 整图，imgsz=1280 |
| 十字交叉点 / 装板（旧功能） | `model/yolo11x-pose-cross-1280-10.pt` | 整图，imgsz=1280 |
| U 型件六角螺套（旧功能） | `model/yolo11l-pose-hexstud-1280-4.pt` | 整图，imgsz=1280 |
| 十字接缝（旧功能） | `model/slot-1280-2.pt` | 整图，imgsz=1280 |

### TensorRT 加速

- 加载模型时优先使用 `model/<权重名>.<指纹>.engine` 缓存（指纹绑定 .pt 内容，换模型自动失效），无缓存则回落 .pt 运行。
- 更换 .pt 后运行 `python3 export_engines.py` 编译（`--check` 只看状态）；**detect 的导出尺寸已按 256 配置**（`model_config.py` 的 `MODEL_EXPORT_IMGSZ`），编译出的 engine 与裁剪推理的静态 shape 一致。
- 预览窗口右上角有**模型状态面板**：每行显示各模型的 ON/OFF 与实时推理耗时，**点击可直接开关模型**（后台热切换，也可通过 socket 的 `model_*_on/off` 控制）。

## 三、启动服务

```bash
cd camera_socket-main
python3 server_socket.py              # 默认即生产配置：方案 0（单钉单拍）
python3 server_socket.py --scheme N   # 指定区域表方案（当前只有方案 0）
```

说明：

- 检测在后台线程持续运行（相机有新帧就推理一次），客户端随时查询拿到的都是**最近一帧**的结果。
- 有图形界面时会弹出实时预览窗口（含模型开关面板）——该窗口仅用于调试，**不要按 q / ESC，会停止检测并结束进程**；headless 环境自动跳过窗口，socket 服务不受影响。
- 相机未连接时服务不退出，自动重连（超过 2 秒没有新帧判定为断开）。
- 默认端口 65432，被占用时启动会报错。
- 每次客户端查询会自动把当前处理后帧 + 结果存入 `datasets/<动作名>/`（数据积累，可关：`server_socket.py` 中 `_save_dataset` 调用）。

## 四、客户端调用（Socket 协议）

- TCP 连接 `服务器IP:65432`
- **请求**：一行**纯文本动作名**，必须以换行符 `\n` 结尾（不是 JSON）
- **响应**：JSON 字符串，自动附带 `timestamp` 字段（接收时间、处理耗时等）
- 服务端响应**不带长度前缀或结束符**；客户端建议累积接收并尝试 `json.loads`，解析成功即为一整条响应

### 动作列表

| 动作 | 说明 | 响应字段 |
|---|---|---|
| `detect_count` | **单钉检测结果（0/1）** | `detect_count` |
| `get_frame` | 获取当前相机画面 | `success` / `frame`(base64) / `shape` |
| `screw_points` | 螺钉关键点坐标（旧功能） | `screw_points_pixels` |
| `cross_points_without_granding` | 十字交叉点（旧功能） | `cross_points_pixels` |
| `board_angle` / `board_position` | 装板角度 / 位置（旧功能） | `board_angle` / `board_position` |
| `u_points` | U 型件六角螺套（旧功能） | `u_points` |
| `slot_points` | 十字接缝位置（旧功能） | `slot_points` |
| `model_{stud,cross,u,slot,detect,board}_on/off` | 动态加载 / 卸载对应模型（释放显存） | `success` / `message` |
| `model_status` | 所有模型的加载与开关状态 | `success` / `models` |

### 客户端示例（Python）

```python
import socket, json

sock = socket.create_connection(('127.0.0.1', 65432))
sock.sendall(b'detect_count\n')

buf = b''
while True:
    buf += sock.recv(4096)
    try:
        resp = json.loads(buf.decode('utf-8'))
        break          # 收到完整的一条响应
    except json.JSONDecodeError:
        continue       # 响应还没收完，继续接收

print(resp['detect_count'])   # 0 或 1
```

### 相机未连接时的行为

以下动作在相机断开时返回提示响应（含一张"相机未连接"的 base64 图片），客户端需按 `success` / `camera_status` 判断，不要直接读结果字段：

`detect_count`、`screw_points`、`cross_points_without_granding`、`board_angle`、`board_position`、`u_points`、`slot_points`、`get_frame`

```json
{"success": false, "frame": "（提示图 base64）", "camera_status": "disconnected", "message": "相机未连接，请等待"}
```

## 五、区域表（stud_detect_regions.json）与标定

区域表定义"裁剪哪里"，服务启动时读取（`regions.py`）。当前只有一个方案（方案 0 = 单钉单拍生产方案）：

```json
{
  "schemes": {
    "0": {
      "margin": 0,
      "conf_threshold": 0.5,
      "imgsz": 256,
      "regions": [ { "id": 1, "box": [972, 451, 1258, 730] } ]
    }
  }
}
```

- `box`：`[x1, y1, x2, y2]` 原图像素坐标——人工标的框，也是模型的全部视野
- `margin`：裁剪外扩像素，单钉方案固定为 `0`
- `conf_threshold`：判定"有钉"的置信度阈值（误检多调高、漏检多调低）
- `imgsz`：推理尺寸，需与模型训练尺寸一致（当前生产模型为 256）
- `schemes` 支持多套方案（多工位标定），通过 `--scheme N` 选择

### 重新标定区域

```bash
python3 annotate_stud_regions.py 一张工位照片.jpg        # 编辑方案 0
python3 annotate_stud_regions.py 一张工位照片.jpg 1      # 编辑 / 新建方案 1
```

操作：鼠标左键拖拽画框（框住螺钉波动范围 + 必要背景），`u` 撤销，`s` 或回车保存。没有工位照片时，可先运行服务后发送 `get_frame` 保存一帧。

### 验证区域框（重要）

```bash
python3 label_tool/check_region_fit.py 照片文件夹 0
```

把一批照片按区域裁剪拼成一张总览图（默认输出 `label_tool/region_fit_overview.jpg`），目检**每张照片的螺钉是否都完整落在框内**。新标定、机械调整后、或怀疑漂移时跑一遍。

## 六、训练数据闭环（迭代模型）

数据集目录不入库（见 `.gitignore`），在本地生成。完整流程：

```bash
# 1. 辅助生成标注：用小模型先框一遍，生成 labelme json（已有的自动跳过）
#    （辅助权重 yolo11x-detect-stud-256-1.pt 放在 model/ 目录，缺失时在训练机上准备）
python3 label_tool/step1_detect_to_labelme.py 照片文件夹 0
labelme 照片文件夹          # 人工审核：删除错框、补画漏框（需安装 labelme）

# 2. 按区域框裁剪训练小图：每张照片固定 1 个样本；
#    有钉照片→正样本，无钉照片→空标注负样本
python3 label_tool/step2_crop_to_yolo.py 照片文件夹 输出文件夹 0

# 3. 平衡正负样本比例（多余的移到 all_removed/ 备份，之后可反向取回）
python3 label_tool/balance_dataset.py 输出文件夹 1     # 1 = 正负 1:1；2 = 1:2

# 4. 用 ultralytics 训练（单类 detect，imgsz=256）后：
#    权重放入 model/，并在 model_config.py 更新 MODEL_PATHS["detect"]，再运行 export_engines.py 编译引擎
```

### 离线验证（不接相机）

```bash
python3 screw_detect_test.py 照片文件夹 [输出文件夹] [方案号]
```

对文件夹照片跑一遍与线上完全一致的检测（暗图 gamma 提亮 → 区域裁剪 → 小图推理 → 每区域最多 1 钉 → 坐标还原），输出标注图：**绿框 = 检出螺钉（含置信度）、红框 = 空位区域、左上角 Count 计数**。用于上线前评估和验收——建议用不同时段/光照的留出集统计**单拍错误率**。

## 七、项目结构

```
camera_socket-main/
├── camera.py                     # 海康相机封装（枚举/连接/取流/断连重连）
├── realtime_test_origin.py       # 检测核心（多模型推理 + 检测线程 + 区域裁剪螺钉检测）
├── server_socket.py              # Socket 服务（对客户端接口）
├── model_config.py               # 模型路径/任务/导出尺寸（唯一定义处）
├── engine_utils.py               # TensorRT engine 缓存管理（指纹绑定）
├── export_engines.py             # TensorRT engine 编译脚本
├── regions.py                    # 螺钉检测区域表加载与裁剪
├── stud_detect_regions.json      # 区域表（方案 0 = 单钉单拍生产方案）
├── screw_detect_test.py          # 离线检测验证脚本
├── annotate_stud_regions.py      # 区域标注工具
├── dataset_saver.py              # 查询触发的历史帧落盘（datasets/）
├── label_tool/
│   ├── step1_detect_to_labelme.py    # 辅助生成 labelme 标注
│   ├── step2_crop_to_yolo.py         # 裁剪训练小图 + 生成 YOLO 标注
│   ├── balance_dataset.py            # 正负样本比例双向平衡
│   └── check_region_fit.py           # 区域适配检查（总览图）
├── lib/                          # 海康相机 SDK 运行库
└── yolov8/                       # 训练辅助脚本
model/                            # 模型权重与 TensorRT engine 缓存（仓库根，不入库）
```

## 常见问题

**启动报"xxx 模型路径不存在"？**
检查仓库根 `model/` 目录是否完整（缺失的模型会告警跳过，对应功能不可用），模型路径由 `model_config.py` 自动定位。

**detect_count 一直是 0？**
逐项检查：1）启动日志是否打印"螺钉检测区域表加载成功"（没有则区域表有问题，此时 detect 直接不检测）；2）用 `model_status` 或预览窗面板确认 detect 模型已加载且开启；3）用 `get_frame` 看画面是否正常；4）用 `screw_detect_test.py` 对同一画面离线比对，区分是模型问题还是相机/配置问题。

**detect 换了新权重后要做什么？**
放入 `model/` → 更新 `model_config.py` 的 `MODEL_PATHS["detect"]` → `python3 export_engines.py --model detect` 编译 256 引擎 → 重启服务（或热替换）。

**预览窗口误按了 q？**
检测线程已停止并会结束整个进程，重启服务即可。

**相机连不上？**
检查 MVS SDK 与 `MVCAM_COMMON_RUNENV`、相机是否被其它程序占用（如 MVS 客户端）、线缆/网络；服务会自动重连。

## 许可证

MIT License
