""" Inference Code """

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule
from src.utils import get_logger

import dotenv
import hydra
from omegaconf import DictConfig
import os
import sys

dotenv.load_dotenv(override=True)
log = get_logger(__name__)

@hydra.main(config_path="configs/", config_name="inference.yaml")
def inference(config: DictConfig):

    # init detector model
    log.info(f"Instantiating detector <{config.detector._target_}>")
    detector = hydra.utils.instantiate(config.detector)

    # init regressor model
    log.info(f"Instantiating regressor <{config.model._target_}>")
    regressor: LightningModule = hydra.utils.instantiate(config.model)
    regressor.load_state_dict(torch.load(config.get('regressor_weights')))
    regressor.eval()

    

def detector_yolov5(model_path: str, cfg_path: str, classes: int, device: str):
    """ YOLOv5 detector model """
    sys.path.append('/raid/didir/Repository/yolo3d-lightning/yolov5')

    # NOTE: ignore import error
    from models.common import AutoShape
    from models.yolo import Model
    from utils.general import intersect_dicts
    from utils.torch_utils import select_device

    device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)
    
    model = Model(cfg_path, ch=3, nc=classes)
    ckpt = torch.load(model_path, map_location=device)  # load
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
    model.load_state_dict(csd, strict=False)  # load
    if len(ckpt['model'].names) == classes:
        model.names = ckpt['model'].names  # set class names attribute
    model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS

    return model.to(device)

if __name__ == '__main__':

    inference()