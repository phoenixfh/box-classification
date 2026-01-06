## 项目说明

生强平台全标注的视野, 识别细胞区域（包含细胞核的 位置），用于给后续识别细胞核路径做准备

因此需要分两步：
- 从视野图识别细胞核区域位置
- 从细胞核小图识别轮廓路径

## 细胞核区域标注数据

- 获取全量标注视野图

注意: 获取的区域稍微扩大一点
```sh
wsictl view get-box --dir=/mnt/ssd/cell-box-data --tars=tars-prod.conf --full --tags=CORE --labels=docs/cell-core/cell.json --storeKey=ai --expand=60

```

数据会生成到 `/mnt/ssd/cell-box-data` , 目录下, 以供训练.

## yolov11训练

识别细胞核的区域

```sh

. ./step_mflow.sh

python yolo/train.py --data docs/cell-core/cell.yaml --model yolo11n.pt --epochs 2000 --imgsz 1024 --batch 16 --single_cls True --project view-cell-box --task_name cell-box-20251225 --device 0,1,2,3,4,5,6,7

python yolo/export_onnx.py  --imgsz 1024  --output view-cell-box/cell-box/weights/best.onnx --model_path view-cell-box/cell-box/weights/best.pt

```


## 细胞核小图数据准备

- 获取全量标注的细胞小图

图片的大小使用 128 

```sh
wsictl view get-unet --dir=/mnt/ssd/cell-core-data --tars=tars-prod.conf --full --tags=CORE --storeKey=ai --expand=128 --type=core
```

说明:
- type: 为 core，表示使用小图训练，每个细胞核对应一张小图， 此时 expand 对应小图的大小，这里为 128*128

数据会生成到 `/mnt/ssd/cell-core-data` , 目录下, 以供训练.

## 使用小图训练

```sh

accelerate launch fastai/train.py --distributed  --data_path /mnt/ssd/cell-core-data --arch unet_seg --img_size 128 --batch_size 512 --epochs 500 --task_name core-20251225 --project_name cell-core-unet

accelerate launch fastai/train.py --distributed  --data_path /mnt/ssd/cell-core-data --arch unet_m_seg --img_size 128 --batch_size 512 --epochs 500 --task_name core-20251225 --project_name cell-core-unet
       


```

## 其他命令

```sh

python fastai/predict.py --model_path runs/cell-core-unet/core/best.pth --arch unet_seg --image /mnt/ssd/cell-core-data/imgs/train/20251027041_09_07.jpg --save_mask

python fastai/export_onnx.py --model runs/cell-core-unet/core/best.pth --data_path /mnt/ssd/cell-core-data --output_path runs/cell-core-unet/core/core-detection.onnx.unet 

python fastai/predict.py --model_path core-detection.onnx.unet --arch unet_seg --image /mnt/ssd/cell-core-data/imgs/train/20251027041_09_07.jpg --save_mask

```
