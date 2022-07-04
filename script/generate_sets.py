"""Create training and validation sets"""

import argparse
from glob import glob
import os


def generate_sets(
    images_path: str, dump_dir: str, postfix: str = "", train_size: float = 0.8
):
    images = glob(os.path.join(images_path, "*.png"))
    ids = [id_.split("/")[-1].split(".")[0] for id_ in images]

    train_sets = sorted(ids[: int(len(ids) * train_size)])
    val_sets = sorted(ids[int(len(ids) * train_size) :])

    for name, sets in zip(["train", "val"], [train_sets, val_sets]):
        name = os.path.join(dump_dir, f"{name}{postfix}.txt")
        with open(name, "w") as f:
            for id in sets:
                f.write(f"{id}\n")

    print(f"[INFO] Training set: {len(train_sets)}")
    print(f"[INFO] Validation set: {len(val_sets)}")
    print(f"[INFO] Total: {len(train_sets) + len(val_sets)}")
    print(f"[INFO] Success Generate Sets")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training and validation sets")
    parser.add_argument("--images_path", type=str, default="../data/KITTI/images")
    parser.add_argument("--dump_dir", type=str, default="../data/KITTI")
    parser.add_argument("--postfix", type=str, default="_80")
    parser.add_argument("--train_size", type=float, default=0.8)
    args = parser.parse_args()

    generate_sets(
        images_path=args.images_path, 
        dump_dir=args.dump_dir, 
        postfix=args.postfix
    )