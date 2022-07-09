"""Generate sets from images folder"""
from glob import glob
import os
import argparse

def generate_sets(images_path: str, dump_dir: str, postfix: str = '', train_size: float = 0.8):
    images = glob(os.path.join(images_path, "*.png"))
    ids = [id_.split('/')[-1].split('.')[0] for id_ in images]

    train_sets = ids[:int(len(ids)*train_size)]
    val_sets = ids[int(len(ids)*train_size):]

    for name, sets in zip(['train', 'val'], [train_sets, val_sets]):
        name = os.path.join(dump_dir, f"{name}{postfix}.txt")
        with open(name, 'w') as f:
            for id in sets:
                f.write(f"{id}\n")

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description="Generate sets from images folder")
    parser.add_argument("--images_path", type=str, default="outputs/images", help="Path to images")
    parser.add_argument("--dump_dir", type=str, default="outputs/sets", help="Path to sets")
    parser.add_argument("--postfix", type=str, default="_", help="Postfix for sets")
    parser.add_argument("--train_size", type=float, default=0.8, help="Train set size")
    args = parser.parse_args()

    # generate sets
    generate_sets(args.images_path, args.dump_dir, args.postfix, args.train_size)