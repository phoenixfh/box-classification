## 项目说明

直接从视野图中识别细胞核轮廓路径

## 视图图标注数据准备

- 获取全量标注视野图

注意: 直接获取的视野图训练 
```sh

wsictl view get-unet --dir=/mnt/ssd/view-core-data --tars=tars-prod.conf --full --tags=CORE --storeKey=ai --type=view
```

说明:
- type: 为 view，表示直接使用视野来训练

数据会生成到 `/mnt/ssd/view-core-data` , 目录下, 以供训练.

## 直接使用视野图训练

```sh

accelerate launch fastai/train.py --distributed  --data_path /mnt/ssd/view-core-data --arch unet_seg --img_size 2048 --batch_size 2 --epochs 500 --task_name view-core --project_name view-core-unet
       
accelerate launch -u fastai/tune.py --config configs/tuning/view-core.yaml --distributed

```

## 其他命令

```sh

python fastai/predict.py --model_path fastai_models/view-core-unet/view-core/best.pth --arch unet_seg --image /mnt/ssd/view-core-data/imgs/train/20251027041_09_07.jpg --save_mask

python fastai/export_onnx.py --model fastai_models/view-core-unet/view-core/best.pth --data_path=/mnt/ssd/view-core-data --output_path=view-core-detection.onnx.unet 

python fastai/predict.py --model_path view-core-detection.onnx.unet --arch unet_seg --image /mnt/ssd/view-core-data/imgs/train/20251027041_09_07.jpg --save_mask

```
