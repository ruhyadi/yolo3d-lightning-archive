# Training Guide

## Detector YOLOv5
Single GPU
```bash
python train.py \
    --img 640 \
    --batch 64 \
    --epochs 3 \
    --data KITTI.yaml \
    --weights yolov5s.pt
```

Multiple GPU
```bash
python -m torch.distributed.launch \
    --nproc_per_node 4 train.py \
    --epochs 20 \
    --batch 64 \
    --data KITTI.yaml \
    --weights yolov5s.pt \
    --device 0,1,2,3
```