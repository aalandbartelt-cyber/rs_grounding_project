#!/usr/bin/env python3
"""校验数据集拷贝完整性：比对源目录与本项目 data 目录下的相对路径、文件大小与抽样 MD5。"""
import hashlib
import os
import sys

SRC = "/mnt/d/上海大学__/26遥感图像文搜图定位挑战赛"
DST = os.path.expanduser("~/rs_grounding_project")

PAIRS = [
    ("训练数据/rs_grounding_train_v1", "data/rs_grounding_train_v1"),
    ("训练数据/rs_grounding_train_v1.zip", "data/rs_grounding_train_v1.zip"),
    ("遥感图像文搜图定位挑战赛（基本信息）.docx",
     "docs/遥感图像文搜图定位挑战赛（基本信息）.docx"),
    ("遥感图像文搜图定位挑战赛（基本信息）.pdf",
     "docs/遥感图像文搜图定位挑战赛（基本信息）.pdf"),
]


def walk(root):
    """返回 {相对路径: 大小}；文件存在 root 下。"""
    out = {}
    if os.path.isfile(root):
        return {"": os.path.getsize(root)}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            out[os.path.relpath(full, root)] = os.path.getsize(full)
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ok = True
    for src_rel, dst_rel in PAIRS:
        src_root = os.path.join(SRC, src_rel)
        dst_root = os.path.join(DST, dst_rel)
        if not os.path.exists(dst_root):
            print(f"[MISSING] {dst_rel}")
            ok = False
            continue
        src_files = walk(src_root)
        dst_files = walk(dst_root)
        missing = sorted(set(src_files) - set(dst_files))
        extra = sorted(set(dst_files) - set(src_files))
        size_diff = [p for p in src_files
                     if p in dst_files and src_files[p] != dst_files[p]]
        total = sum(src_files.values())
        status = "OK" if not (missing or extra or size_diff) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] {dst_rel}: {len(src_files)} files, "
              f"{total / 1048576:.1f} MiB, "
              f"missing={len(missing)} extra={len(extra)} size_diff={len(size_diff)}")
        for p in size_diff[:5]:
            print(f"    size mismatch: {p} {src_files[p]} != {dst_files[p]}")

    # 抽样 MD5：每类各取一个文件
    samples = [
        ("训练数据/rs_grounding_train_v1/train/images", "data/rs_grounding_train_v1/train/images"),
        ("训练数据/rs_grounding_train_v1/train/labels_json",
         "data/rs_grounding_train_v1/train/labels_json"),
        ("训练数据/rs_grounding_train_v1/val/labels_xml",
         "data/rs_grounding_train_v1/val/labels_xml"),
    ]
    print("\n-- MD5 spot check --")
    for src_rel, dst_rel in samples:
        src_dir = os.path.join(SRC, src_rel)
        dst_dir = os.path.join(DST, dst_rel)
        name = sorted(os.listdir(src_dir))[0]
        a = md5(os.path.join(src_dir, name))
        b = md5(os.path.join(dst_dir, name))
        print(f"{'OK ' if a == b else 'BAD'} {name} {a[:12]}")
        ok = ok and a == b

    print("\nRESULT:", "ALL OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
