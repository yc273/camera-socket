#!/usr/bin/env python3
"""
YOLO11 cross 模型推理测试脚本
忠实复刻 camera_socket-main/realtime_test_origin.py 的模型加载与推理用法，
验证 model/yolo11-cross-x-1280-8.pt 能否在 test-datasets/images-cross-1/ 上正常推理。

用法（在 camera conda 环境、工作区根目录下）:
    python test_model.py                 # 跑全部图片
    python test_model.py --limit 10      # 只跑前 10 张（冒烟测试）
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLO11 cross 模型推理测试")
    p.add_argument("--model", default="model/yolo11-cross-x-1280-8.pt",
                   help="模型权重路径")
    p.add_argument("--imgdir", default="test-datasets/images-cross-1",
                   help="测试图片目录")
    p.add_argument("--limit", type=int, default=0,
                   help="只跑前 N 张图片（0=全部）")
    p.add_argument("--warmup-imgsz", type=int, default=640,
                   help="预热 imgsz（对齐源码 line 163）")
    p.add_argument("--imgsz", type=int, default=1280, help="推理 imgsz（对齐源码 line 557）")
    p.add_argument("--max-det", type=int, default=2, help="每张图最多检测数（对齐源码 max_det=2）")
    p.add_argument("--vis-dir", default="test_output/vis",
                   help="可视化结果保存目录（画好关键点/框的图片，肉眼检验用）")
    p.add_argument("--no-vis", action="store_true", help="不保存可视化图片")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- GPU 环境确认 ----
    print("=" * 60)
    print("环境信息")
    print("=" * 60)
    print(f"torch            : {torch.__version__}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device      : {torch.cuda.get_device_name(0)}")
    print(f"ultralytics      : {YOLO().__class__.__module__}")

    # ---- 收集图片 ----
    images = sorted(glob.glob(os.path.join(args.imgdir, "*.jpg")))
    if args.limit > 0:
        images = images[:args.limit]
    print(f"\n测试图片数      : {len(images)}  (目录: {args.imgdir})")
    print(f"模型            : {args.model}")
    if len(images) == 0:
        print("未找到任何 .jpg 图片，退出。")
        return

    # ---- 可视化输出目录 ----
    save_vis = not args.no_vis
    if save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f"可视化输出目录  : {args.vis_dir}")

    # ---- 加载模型 ----
    print("\n" + "=" * 60)
    print("加载模型 ...")
    print("=" * 60)
    model = YOLO(args.model)

    # ---- 预热（对齐源码 _load_model_to_gpu line 163）----
    print("预热模型 ...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model.predict(source=dummy, imgsz=args.warmup_imgsz, conf=0.5,
                      verbose=False, save=False, show=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("模型预热完成\n")

    # ---- 推理 ----
    print("=" * 60)
    print("开始推理")
    print("=" * 60)
    times = []                 # 每张图推理耗时(s)
    n_ok = 0                   # 成功推理张数
    n_err = 0                  # 失败张数
    total_points = 0           # 检出关键点总数
    point_count_dist = {}      # {每张图点数: 张数}
    samples = []               # 前 10 张的样例结果
    errors = []                # 错误详情

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            n_err += 1
            errors.append((os.path.basename(img_path), "cv2.imread 失败"))
            continue

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # stream=False：需要 result 对象画图（stream=True 生成器会消耗掉）
            results = model.predict(
                source=img,
                imgsz=args.imgsz,
                conf=0,
                save=False,
                show=False,
                verbose=False,
                stream=False,
                max_det=args.max_det,
            )

            # 解析 keypoints（对齐源码 _parse_detection_results_with_confidence line 720-741）
            points = []
            for result in results:
                kpts = result.keypoints
                if kpts is not None and kpts.xy is not None:
                    for kpt in kpts.xy:
                        try:
                            x, y = kpt[0]
                            points.append((round(float(x), 2), round(float(y), 2)))
                        except Exception:
                            continue

                # 保存可视化图片（ultralytics 内置，自动画关键点/骨架/框+置信度）
                if save_vis:
                    plotted = result.plot()  # 返回 BGR numpy 数组
                    out_name = os.path.basename(img_path)
                    cv2.imwrite(os.path.join(args.vis_dir, out_name), plotted)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        except Exception as e:
            n_err += 1
            errors.append((os.path.basename(img_path), repr(e)))
            continue

        times.append(dt)
        n_ok += 1
        total_points += len(points)
        point_count_dist[len(points)] = point_count_dist.get(len(points), 0) + 1
        if i < 10:
            samples.append((os.path.basename(img_path), points))

        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            avg_so_far = sum(times) / len(times) * 1000 if times else 0
            print(f"  进度 {i + 1}/{len(images)}  累计平均 {avg_so_far:.1f} ms/张")

    # ---- 汇总报告 ----
    print("\n" + "=" * 60)
    print("推理结果汇总")
    print("=" * 60)
    print(f"总图片数         : {len(images)}")
    print(f"成功推理         : {n_ok}")
    print(f"失败             : {n_err}")
    if n_ok > 0:
        print(f"总推理耗时       : {sum(times):.2f} s")
        print(f"平均推理耗时     : {sum(times) / n_ok * 1000:.1f} ms/张")
        print(f"最快 / 最慢      : {min(times) * 1000:.1f} / {max(times) * 1000:.1f} ms")
        print(f"平均检出关键点数 : {total_points / n_ok:.2f} 个/张")
        print(f"每张图点数分布   : {dict(sorted(point_count_dist.items()))}")
        n_nonzero = sum(c for k, c in point_count_dist.items() if k > 0)
        print(f"有检出图片占比   : {n_nonzero}/{n_ok} ({n_nonzero / n_ok * 100:.1f}%)")

    if samples:
        print("\n前 10 张样例（文件名 -> 检出的交叉点坐标）:")
        for name, pts in samples:
            print(f"  {name}: {pts}")

    if errors:
        print("\n错误详情:")
        for name, msg in errors[:20]:
            print(f"  {name}: {msg}")
        if len(errors) > 20:
            print(f"  ... 另有 {len(errors) - 20} 条错误未显示")

    print("\n结论:", "✅ 模型可正常运行" if n_err == 0 and n_ok > 0 else "⚠️ 存在错误，见上方详情")
    if save_vis and n_ok > 0:
        print(f"\n👉 可视化图片已保存到: {os.path.abspath(args.vis_dir)}")
        print("   用文件管理器打开该目录，肉眼对比原图检验关键点位置是否准确。")


if __name__ == "__main__":
    main()
