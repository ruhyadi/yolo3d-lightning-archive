""" Inference Code """

from typing import List
from PIL import Image
import cv2
from glob import glob
import numpy as np

import torch
from torchvision.transforms import transforms
from pytorch_lightning import LightningModule
from src.utils import Calib, get_logger
from src.datamodules.components.kitti_dataset import DetectedObject
from src.utils.ClassAverages import ClassAverages
from src.utils.Plotting import plot_3d_box
from src.utils.Math import calc_location
from src.utils.Plotting import calc_theta_ray


import dotenv
import hydra
from omegaconf import DictConfig
import os
import sys

dotenv.load_dotenv(override=True)
log = get_logger(__name__)


@hydra.main(config_path="configs/", config_name="inference.yaml")
def inference(config: DictConfig):

    # use global calib file
    proj_matrix = Calib.get_P(config.get("calib_file"))

    # Averages Dimension list
    class_averages = ClassAverages()

    # angle bins
    angle_bins = generate_bins(bins=2)

    # init detector model
    log.info(f"Instantiating detector <{config.detector._target_}>")
    detector = hydra.utils.instantiate(config.detector)

    # init regressor model
    log.info(f"Instantiating regressor <{config.model._target_}>")
    regressor: LightningModule = hydra.utils.instantiate(config.model)
    regressor.load_state_dict(torch.load(config.get("regressor_weights")))
    regressor.eval()

    # init preprocessing
    log.info(f"Instantiating preprocessing")
    preprocess: List[torch.nn.Module] = []
    if "augmentation" in config:
        for _, conf in config.augmentation.items():
            if "_target_" in conf:
                preprocess.append(hydra.utils.instantiate(conf))
    preprocess = transforms.Compose(preprocess)

    # TODO: able inference on videos
    imgs_path = sorted(glob(os.path.join(config.get("source_dir"), "*")))
    for img_path in imgs_path:
        name = img_path.split("/")[-1].split(".")[0]
        img = Image.open(img_path)
        img_draw = np.asarray(img)
        # detect object with Detector
        dets = detector(img).crop()

        DIMS = []

        for i, det in enumerate(dets):
            # preprocess img with torch.transforms
            crop = preprocess(cv2.resize(det["im"], (224, 224)))
            # batching img
            crop = crop.reshape((1, *crop.shape))
            # regress 2D bbox
            [orient, conf, dim] = regressor(crop)
            orient = orient.detach().numpy()[0, :, :]
            conf = conf.detach().numpy()[0, :]
            dim = dim.detach().numpy()[0, :]
            # refinement dimension
            try:
                dim += class_averages.get_item(class_to_labels(det["cls"].numpy()))
                DIMS.append(dim)
            except:
                dim = DIMS[-1]
            # calculate theta ray
            theta_ray = calc_theta_ray(img.size[0], det["box"], proj_matrix)
            # box_2d
            box_xyxy = [x.numpy() for x in det['box']]
            # TODO: understand this
            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi
            # calculate the location
            location, x = calc_location(
                dimension=dim,
                proj_matrix=proj_matrix,
                box_2d=box_xyxy,
                alpha=alpha,
                theta_ray=theta_ray
            )
            # orientation
            orient = alpha + theta_ray
            # plot 3d bbox
            plot_3d_box(
                img=img_draw, 
                cam_to_img=proj_matrix, 
                ry=orient, 
                dimension=dim, 
                center=location
            )

        if config.get("save_result"):
            cv2.imwrite(f'{config.get("dump_dir")}/{name}_{i:03d}.png', img_draw)

def detector_yolov5(model_path: str, cfg_path: str, classes: int, device: str):
    """YOLOv5 detector model"""
    sys.path.append("/raid/didir/Repository/yolo3d-lightning/yolov5")

    # NOTE: ignore import error
    from models.common import AutoShape
    from models.yolo import Model
    from utils.general import intersect_dicts
    from utils.torch_utils import select_device

    device = select_device(
        ("0" if torch.cuda.is_available() else "cpu") if device is None else device
    )

    model = Model(cfg_path, ch=3, nc=classes)
    ckpt = torch.load(model_path, map_location=device)  # load
    csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=["anchors"])  # intersect
    model.load_state_dict(csd, strict=False)  # load
    if len(ckpt["model"].names) == classes:
        model.names = ckpt["model"].names  # set class names attribute
    model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS

    return model.to(device)


def class_to_labels(class_: int, list_labels: List = None):

    if list_labels is None:
        list_labels = ["pedestrian", "car", "truck"]

    return list_labels[int(class_)]

def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of bins

    return angle_bins


if __name__ == "__main__":

    inference()
