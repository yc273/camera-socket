"""训练集正负样本比例平衡工具（双向）：
统计数据集中正样本（txt 非空）和负样本（txt 空）的数量，按预设正负比自动调整：
负样本多余目标 → 随机把多余的（jpg+txt 成对）移动到备份文件夹 all_removed/；
负样本少于目标 → 从备份随机取回补足。1:1 ↔ 1:2 等比例反复调整均可，物理文件保留。

背景：负样本比例过大时会稀释训练注意力，一般正负 1:1 或 1:2 起步。

用法（在 label_tool 或项目根运行均可）：
    python label_tool/balance_dataset.py                处理 label_tool/all，正负比 1:1
    python label_tool/balance_dataset.py 某数据集目录    指定数据集目录
    python label_tool/balance_dataset.py 某数据集目录 2  正负比 1:2（每个正样本配2个负样本）
"""
import os
import sys
import random
import shutil

# 随机抽样种子（None=每次结果不同；设为整数可复现）
SEED = None

# 数据集目录默认取本脚本同级的 all（不受运行时工作目录影响）
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all")
# 负样本备份目录（移出/取回都在这里，成对 jpg+txt）
REMOVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_removed")


def classify_samples(labels_dir):
    """按 txt 是否非空分类样本，返回 (正样本stem列表, 负样本stem列表, 无对应txt的孤立jpg数)"""
    positive = []
    negative = []
    for name in os.listdir(labels_dir):
        if not name.endswith('.txt'):
            continue
        stem = os.path.splitext(name)[0]
        txt_path = os.path.join(labels_dir, name)
        if os.path.getsize(txt_path) > 0:
            positive.append(stem)
        else:
            negative.append(stem)
    orphans = 0
    images_dir = os.path.join(os.path.dirname(labels_dir), "images")
    if os.path.isdir(images_dir):
        for name in os.listdir(images_dir):
            stem = os.path.splitext(name)[0]
            if not os.path.exists(os.path.join(labels_dir, stem + '.txt')):
                orphans += 1
    return positive, negative, orphans


def move_sample(stem, src_root, dst_root):
    """把一个样本的 jpg + txt 成对从 src_root 移动到 dst_root（某一半不存在则只移另一半）"""
    for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
        src = os.path.join(src_root, sub, stem + ext)
        if os.path.exists(src):
            dst_dir = os.path.join(dst_root, sub)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, os.path.join(dst_dir, stem + ext))


def collect_backup_negatives(labels_dir):
    """收集备份目录中可取回的负样本 stem：
    txt 存在且为空（确是负样本）、images 里有同名 jpg（成对）、数据集里无同名文件（防重名）"""
    backup_labels = os.path.join(REMOVED_DIR, "labels")
    backup_images = os.path.join(REMOVED_DIR, "images")
    dataset_labels = labels_dir
    dataset_images = os.path.join(os.path.dirname(labels_dir), "images")
    available = []
    if not os.path.isdir(backup_labels):
        return available
    for name in os.listdir(backup_labels):
        if not name.endswith('.txt'):
            continue
        stem = os.path.splitext(name)[0]
        txt_path = os.path.join(backup_labels, name)
        if os.path.getsize(txt_path) > 0:
            continue
        if not os.path.exists(os.path.join(backup_images, stem + '.jpg')):
            continue
        if (os.path.exists(os.path.join(dataset_labels, stem + '.txt'))
                or os.path.exists(os.path.join(dataset_images, stem + '.jpg'))):
            continue
        available.append(stem)
    return available


def main():
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_DIR
    ratio = 1.0
    if len(sys.argv) > 2:
        try:
            ratio = float(sys.argv[2])
        except ValueError:
            print(f"正负比 '{sys.argv[2]}' 不是数字")
            sys.exit(1)

    labels_dir = os.path.join(dataset_dir, "labels")
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
        print(f"错误：{dataset_dir} 下需要 images/ 和 labels/ 两个子目录")
        sys.exit(1)

    positive, negative, orphans = classify_samples(labels_dir)
    print(f"数据集：{dataset_dir}")
    if positive:
        print(f"当前样本：正样本 {len(positive)} | 负样本 {len(negative)} | "
              f"正负比 1:{len(negative) / len(positive):.2f}")
    else:
        print(f"当前样本：正样本 0 | 负样本 {len(negative)}")
    if orphans:
        print(f"警告：发现 {orphans} 张没有对应 txt 的孤立图片，已跳过不处理")

    keep_count = int(round(len(positive) * ratio))
    if SEED is not None:
        random.seed(SEED)

    if len(negative) > keep_count:
        # 负样本多余目标：随机移出
        to_remove = random.sample(negative, len(negative) - keep_count)
        for stem in to_remove:
            move_sample(stem, dataset_dir, REMOVED_DIR)
        print(f"目标正负比 1:{ratio:g} → 移出 {len(to_remove)} 个多余负样本到 {REMOVED_DIR}")
    elif len(negative) < keep_count:
        # 负样本少于目标：从备份取回
        need = keep_count - len(negative)
        available = collect_backup_negatives(labels_dir)
        if not available:
            print(f"目标正负比 1:{ratio:g} → 需补 {need} 个负样本，但备份目录 {REMOVED_DIR} 中没有可取回的负样本")
            sys.exit(1)
        to_restore = random.sample(available, min(need, len(available)))
        for stem in to_restore:
            move_sample(stem, REMOVED_DIR, dataset_dir)
        print(f"目标正负比 1:{ratio:g} → 从备份取回 {len(to_restore)} 个负样本")
        if len(to_restore) < need:
            print(f"警告：备份数量不足，仍差 {need - len(to_restore)} 个负样本（总负样本上限受历史移出量限制）")
    else:
        print(f"目标正负比 1:{ratio:g} → 已满足，无需处理")
        return

    # 重新统计打印最终分布
    positive, negative, _ = classify_samples(labels_dir)
    if positive:
        print(f"最终样本：正样本 {len(positive)} | 负样本 {len(negative)} | 正负比 1:{len(negative) / len(positive):.2f}")


if __name__ == "__main__":
    main()
