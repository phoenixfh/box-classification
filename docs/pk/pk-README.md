## 项目说明

PK 玻片识别阳性

## 训练数据准备

- 获取全量标注视野图
```sh
wsictl view get-box --dir=/mnt/ssd/pk-box-data --tars=tars-test.conf --tags=PK --labels=docs/pk/pk-rect.json --storeKey=ai
wsictl view get-box --dir=/mnt/ssd/pk-box-data --tars=tars-prod.conf --tags=PK --labels=docs/pk/pk-rect.json --storeKey=ai

wsictl view get-box --dir=/mnt/ssd/pk-box-data --tars=tars-prod.conf --barCode=20250625018 --labels=docs/pk/pk-rect.json --storeKey=ai
wsictl view get-box --dir=/mnt/ssd/pk-box-data --tars=tars-prod.conf --barCode=20250625016 --labels=docs/pk/pk-rect.json --storeKey=ai
wsictl view get-box --dir=/mnt/ssd/pk-box-data --tars=tars-prod.conf --barCode=20250625013 --labels=docs/pk/pk-rect.json --storeKey=ai
wsictl view get-box --dir=/mnt/ssd/pk-box-data --tars=tars-prod.conf --barCode=20250625005 --labels=docs/pk/pk-rect.json --storeKey=ai


```

数据会生成到 `/mnt/ssd/pk-box-data`, 目录下, 以供训练.

## 下载 yolov5的代码

```sh
conda create -n yolov5 python=3.10
conda activate yolov5
git clone https://github.com/ultralytics/yolov5
```

## 训练过程

```sh
conda activate yolov5
export PYTHONWARNINGS="ignore::FutureWarning"
python yolov5/train.py --img 640 --epochs 200 --data docs/pk/pk-dataset.yaml --weights yolov5s.pt
```

## Detect

```sh
python yolov5/detect.py --weights yolov5/runs/train/pk-positive/weights/best.pt  --img 640 --source /mnt/ssd/pk-box-data/images/val/

```

## Export

```sh
python yolov5/export.py --weights yolov5/runs/train/pk-positive/weights/best.pt --include onnx
```
