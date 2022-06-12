""" Inference Code """

from typing import List
from PIL import Image
import cv2
from glob import glob

import torch
from torchvision.transforms import transforms
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

    # init preprocessing
    log.info(f"Instantiating preprocessing")
    preprocess: List[torch.nn.Module] = []
    if 'augmentation' in config:
        for _, conf in config.augmentation.items():
            if '_target_' in conf:
                preprocess.append(hydra.utils.instantiate(conf))
    preprocess = transforms.Compose(preprocess)
    
    # single image
    # TODO: able inference on sequence images
    imgs_path = sorted(glob(os.path.join(config.get('source_dir'), '*')))
    for img_path in imgs_path:
        name = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path)
        # detect 2D bounding box
        dets = [cv2.resize(det['im'], (224, 224), interpolation=cv2.INTER_CUBIC) \
                for det in detector(img).crop()]
        for i, det in enumerate(dets):
            # preprocess img
            det = preprocess(det)
            # batching img
            det = det.reshape((1, *det.shape))
            res = regressor(det)
            print(res)

            if config.get('save_det2d'):
                cv2.imwrite(f'{config.get("dump_dir")}/{name}_{i:03d}.png', det)

    dets = detector(img).crop()
    for i, det in enumerate(dets):
        cv2.imwrite(f'/raid/didir/Repository/yolo3d-lightning/runs/sample_{i}.png', det['im'])

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