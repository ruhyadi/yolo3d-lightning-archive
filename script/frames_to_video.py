"""Generate frames to vid"""

import argparse
import cv2
from glob import glob
import os

def generate(imgs_path, vid_path, fps=30, frame_size=(640, 480)):
    """Generate frames to vid"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_writer = cv2.VideoWriter(vid_path, fourcc, fps, frame_size)
    imgs_glob = sorted(glob(os.path.join(imgs_path, "*")))
    for img_path in imgs_glob:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        vid_writer.write(img)
    vid_writer.release()
    print('[INFO] Video saved to {}'.format(vid_path))

if __name__ == "__main__":
    # create argparser
    parser = argparse.ArgumentParser(description="Generate frames to vid")
    parser.add_argument("--imgs_path", type=str, default="runs", help="path to imgs")
    parser.add_argument("--vid_path", type=str, default="outputs/videos/001.mp4", help="path to vid")
    parser.add_argument("--fps", type=int, default=30, help="fps")
    parser.add_argument("--frame_size", type=int, nargs=2, default=(1242, 375), help="frame size")
    args = parser.parse_args()

    # generate vid
    generate(args.imgs_path, args.vid_path, args.fps, args.frame_size)