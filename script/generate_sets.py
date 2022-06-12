
import numpy as np

from glob import glob
import os

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
    
    generate_sets(
        images_path='/raid/didir/Repository/yolo3d-lightning/data/KITTI/training/images',
        dump_dir='/raid/didir/Repository/yolo3d-lightning/data/KITTI/training',
        postfix='_80', 
        train_size=0.8)