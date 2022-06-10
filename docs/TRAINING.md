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

## Regressor
Muiltiple GPU
```bash
python train.py \
    trainer.gpus=[1,2,3] \
    trainer.max_epochs=20 \
    +trainer.strategy=ddp \
    name="20 Epochs"
```

#### Debugging
Train with 5% of data
```bash
python train.py \
    +trainer.limit_train_batches=0.05 \
    +trainer.limit_val_batches=0.05 \
    name=test5%-data
```

Multiple GPU
```bash
python -m torch.distributed.launch \
    --nproc_per_node 2 \
    train.py \
        --batch 64 \
        --data coco.yaml \
        --weights yolov5s.pt \
        --device 0,1
```