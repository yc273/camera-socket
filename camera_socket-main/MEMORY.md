# 项目记忆

## 项目是什么
海康工业相机实时视觉检测服务（Jetson + Linux）：相机取流 → 多个 YOLO 模型推理（TensorRT 加速）→ TCP Socket 服务供客户端调用。

## 架构（数据流）
1. `camera.py`：海康 MVS SDK 封装。枚举/打开设备、开启自动曝光增益、后台线程抓帧，支持 Mono8 / BayerRG8（转 RGB）。依赖 `MvCameraControl_class.py` 等官方 SDK 绑定；Linux 需装 MVS SDK 并设置 `MVCAM_COMMON_RUNENV`。
2. `realtime_test_origin.py`：核心。`RealTimePoseDetector` 线程安全单例，模块底部创建全局 `detector`。
   - 5 个模型：stud 螺钉关键点(pose)、cross 十字交叉点(pose，board 装板检测共用同一模型文件)、u U型件六角螺套(pose)、slot 十字接缝(pose)、detect 螺钉检测框(**单钉单拍生产**)。各模型有 loaded/enabled 两个状态，可经 socket 或预览窗面板动态加载/卸载。
   - **TensorRT**：加载时优先命中 `model/<权重>.<指纹>.engine` 缓存（`engine_utils.py`，指纹绑定 .pt 内容）；无缓存回落 .pt。换权重后跑 `export_engines.py` 编译。加载模型必须显式传 task（`model_config.MODEL_TASKS`），否则 slot 这种文件名没关键字的会被 ultralytics 猜成 detect 导致关键点解析失效。
   - 检测线程后台跑（新帧事件触发），暗图自动 gamma 2.2 提亮（阈值 80）；相机断连自动重连（2 秒无帧判定）；headless 环境自动跳过预览窗。
   - 预览窗右上角模型状态面板：ON/OFF + 实时耗时，点击热切换（2026-10-15 加入）。
   - 共享帧通过模块级 `get_shared_frame()/set_shared_frame()` 供外部取图；`get_last_frames()` 供 dataset_saver 落盘。
3. **螺钉检测 ROI 裁剪方案（生产）**：人工标定"放螺钉"的矩形区域到 `stud_detect_regions.json`（`regions.py` 加载/裁剪），检测时裁剪小图推理 `yolo26x-detect-stud-256-2.pt`（imgsz=256），每区域严格最多 1 个钉（取最高置信度框，坐标还原回整图），`detect_count` = 框数。
   - **单钉单拍方案（2026-10-12 敲定为生产）**：旧方案"一张图 40 个位"整图错误率最高 10%（折算单钉错率约 0.26%）工业不可接受，改为**一次拍一颗钉 + 单区域方案**（钉每次基本落在同一区域、同一姿态）。**margin=0，人工框就是唯一裁剪窗口**：训练（step2 ENLARGE_RATIO=0）与推理（方案 margin=0）用同一个框，天然一致。detect_count 变 0/1。json 只保留方案 0；主服务默认即生产配置（`--scheme` 默认 0）。
   - **区域表加载失败时 detect 直接不检测，无整图兜底**（256 小图模型跑整图必然大量误检漏检；整图兜底已删）。
   - **detect 的 engine 导出尺寸必须是 256**（`model_config.MODEL_EXPORT_IMGSZ`），按默认 1280 编译会与裁剪推理静态 shape 不匹配。
4. `annotate_stud_regions.py`：区域标注工具。鼠标拖框（u 撤销、s 保存），可指定图片和方案号；新方案默认 margin=0；OUTPUT_JSON 锚定脚本目录，不受工作目录影响。
5. `server_socket.py`：TCP 服务，默认 **0.0.0.0:65432**。协议：客户端发一行**纯文本动作名**（\n 结尾，非 JSON），服务端回 JSON（**无长度前缀/结束符**，客户端累积接收至 json.loads 成功）。动作：detect_count / screw_points / cross_points_without_granding / board_angle / board_position / u_points / slot_points / get_frame / model_status / model_{stud,cross,u,slot,detect,board}_on|off。相机断连时相关动作返回提示帧（含 detect_count，防读到过期结果）；每次查询自动把处理后帧+结果存 `datasets/<动作名>/`（dataset_saver）。
6. `label_tool/` 数据集工具链（脚本自带路径注入，可在任意工作目录运行）：
   - `step1_detect_to_labelme.py`：区域裁剪+小模型辅助生成 labelme 标注（辅助权重 yolo11x-detect-stud-256-1.pt，训练机用）。用法 `python label_tool/step1_detect_to_labelme.py 图片夹 [方案号]` → labelme 审核
   - `step2_crop_to_yolo.py`：按区域表裁剪训练小图，**ENLARGE_RATIO=0（小图 = 人工框本身）**；每窗口最多 1 钉；无钉窗口生成空 txt 负样本；**单区域方案下"json 有框但都不在窗口内"属异常，警告+跳过**；输出 images/+labels/，命名 {原图名}_r{区域id}
   - `balance_dataset.py`：正负样本双向平衡。用法 `python label_tool/balance_dataset.py [数据集目录] [正负比]`
   - `check_region_fit.py`：区域适配检查——批量裁出方案窗口拼总览图，目检"螺钉完整落在框内"；画框迭代/上线验证/防机械漂移抽检，不需模型。用法 `python label_tool/check_region_fit.py 图片夹 [方案号] [输出图]`
   - `screw_detect_test.py`：离线验证——对照片文件夹跑与线上一致的全流程（gamma→裁剪→推理→还原），绿框检出/红框空位/Count 计数；模型路径取 `model_config.MODEL_PATHS["detect"]`
   - 数据流程：step1 辅助标注 → labelme 审核 → step2 裁剪 → balance_dataset 平衡 → 训练
7. 数据集、`test_result/` 等工作数据均 gitignore 不入库。

## 当前状态（2026-10-15）
- **单钉单拍方案已从 Windows 代码线（camera_socket_code_20261015 快照）移植进本目录并验证**：区域表唯一方案 0（单区域 [972,451,1258,730]，margin=0 / conf=0.5 / imgsz=256）；detect 整图推理旧分支已删除；离线冒烟（空区域 0 / 亮斑 1、坐标还原正确）与全模型加载集成检查通过。快照目录移植完即删除。
- 模型统一放仓库根 `model/`（`model_config.py` 自动定位）。**生产检测模型 = model/yolo26x-detect-stud-256-2.pt**（无 engine 缓存，按 .pt 运行，单区域约 90ms；需要加速跑 `export_engines.py --model detect`）。其余 pose/1280 模型为旧功能保留。
- 验收指标：用不同时段/光照留出集统计**单拍错误率**（旧方案折算单钉错率约 0.26%，先测基线再定目标）。
- 环境：Jetson 系统 python3，ultralytics 8.4.26 + torch 2.10 + CUDA；Windows 侧训练用 conda yolov8 环境（requirements.txt 各自独立，勿互拷）。
- 历史经验：①裁剪窗口训练/推理必须一致（实测 margin 不匹配：同一张图 margin=0 检出 8 颗、margin=100 只检出 1 颗）——单钉单拍已用"两边都是人工框"根治；②侧斜姿态的钉可能模型 0 响应（姿态覆盖是错率主因，单钉单拍就是为此而来）；③框与窗口交集面积必须先 clamp 到 ≥0 再相乘（曾致"窗口内多个钉"误判）；④`run()` 会弹 OpenCV 实时显示窗口，socket 服务启动后也会弹出；headless 无窗口但服务正常；⑤多区域方案 + detect 静态 batch=1 engine 时批量裁剪推理会 shape 不匹配，需改逐区域推理（当前单区域不触发）。
