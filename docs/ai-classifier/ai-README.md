## 项目说明

生强平台全标注的视野, 训练分割视野图, 分割成单个细胞核团聚细胞, 然后使用小图分类


## 切图数据准备

- 获取全量标注视野图
两种方式: 不区分单个和团, 区分单个和团

```sh
wsictl view get-box --dir=/mnt/ssd/ai-detection-data --tars=tars-test.conf --full --tags=AIM,AIC,CUTAIC --labels=docs/ai-classifier/ai-detection.json --storeKey=ai

wsictl view get-box --dir=/mnt/ssd/ai-detection-data --tars=tars-prod.conf --full --tags=AIM,AIC,CUTAIC --labels=docs/ai-classifier/ai-detection.json --storeKey=ai

```

数据会生成到 `/mnt/ssd/ai-detection-data` , 目录下, 以供训练.

### 使用 yolov11 做识别
```sh

```sh


python yolo/train.py --data docs/ai-classifier/ai-detection.yaml --model yolo11l.pt --epochs 1000 --imgsz 640 --batch 4 --single_cls True --project ai-detection-yolo --task_name detection.20251126 --device 0,1,2,3,4,5,6,7

python yolo/export_onnx.py  --imgsz 640  --output runs/ai-detection-yolo/detection.20251126/weights/best.onnx --model_path runs/ai-detection-yolo/detection.20251126/weights/best.pt

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master-port 33333 --nproc_per_node=4 yolo/tune.py --config configs/tuning/ai-detection.yaml --distributed


```

## 小图拉取

- 获取标签数据
```sh

wsictl view get-box-image --dir=/mnt/ssd/ai-classify-data --tars=tars-prod.conf --tags=AIM,AIC --labels=docs/ai-classifier/ai-classify.json --storeKey=ai

wsictl view get-box-image --dir=/mnt/ssd/ai-classify-data --tars=tars-prod.conf --tags=AIM,AIC --labels=docs/ai-classifier/ai-classify.json --storeKey=ai --local

wsictl view get-box-image --dir=/mnt/ssd/ai-classify-all-data --tars=tars-prod.conf --tags=AIM,AIC --labels=docs/ai-classifier/ai-classify-all.json --storeKey=ai

wsictl view get-box-image --dir=/mnt/ssd/ai-classify-all-data --tars=tars-prod.conf --tags=AIM,AIC --labels=docs/ai-classifier/ai-classify-all.json --storeKey=ai --local
```


### 使用 yolov11做分类训练
```sh

yolo setting mlflow=True

# 效果比较好， lr0得值需要根据 batch size 重点设置
export MLFLOW_TRACKING_URI=http://192.168.16.130:5000
export AWS_ACCESS_KEY_ID=mlflow
export AWS_SECRET_ACCESS_KEY=mlflow@SN
export AWS_ENDPOINT_URL=http://192.168.16.130:9000
export MLFLOW_S3_IGNORE_TLS=true

yolo classify train data=/mnt/ssd/ai-classify-all-data model=yolo11s-cls.pt imgsz=224 batch=1024 epochs=1000 project=ai-classifier-yolo name=yolov11.20260102  device=0,1,2,3,4,5,6,7 patience=50 cos_lr=True lr0=0.01 lrf=0.1 optimizer=AdamW


yolo classify train data=/mnt/ssd/ai-classify-data model=yolo11s-cls.pt imgsz=224 batch=1024 epochs=1000 project=ai-classifier-yolo name=yolov11-20251126 device=0,1,2,3,4,5,6,7

yolo classify train data=/mnt/ssd/ai-classify-all-data model=yolo11s-cls.pt imgsz=224 batch=1024 epochs=1000 project=ai-classifier-yolo name=yolov11.20251128.all device=4,5,6,7


python yolo/export_onnx.py --model_path ai-classifier-yolo/yolov11.202512023/weights/best.pt --imgsz=224 --output=ai-classifier-yolo/yolov11.202512023/weights/best.onnx

# yolo classify train data=/mnt/ssd/ai-classify-data model=yolo11s-cls.pt imgsz=224 batch=32 name=ai.20251010 epochs=500 project=ai-classifier name=yolov11

# yolo classify train data=/mnt/ssd/ai-classify-data model=runs/classify/ai.20251010/weights/best.pt epochs=500 imgsz=224 batch=32 name=ai.20251010

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=4 fastai/train.py --distributed  --data_path /mnt/ssd/ai-classify-data --arch yolov11s_cls --img_size 224 --batch_size 1024 --epochs 1000 --task_name yolov11.20251127 --project_name ai-classifier-yolo
       
# yolo classify predict model=runs/classify/ai.202509222/weights/best.pt source=/mnt/ssd/dna-classify-cluster-data/val/


### 使用fastai来训练

```sh
export PYTHONWARNINGS="ignore::FutureWarning"

accelerate launch fastai/train.py --distributed  --data_path /mnt/ssd/ai-classify-all-data --img_size=224 --batch_size=256 --arch=resnet18 --epochs=500 --project_name=ai-classifier-resnet18 --task_name=resnet18-20251205 --train_size=10000 --val_size=1000

accelerate launch fastai/train.py --distributed  --data_path /mnt/ssd/ai-classify-all-data --img_size=224 --batch_size=256 --arch=resnet18 --epochs=500 --project_name=ai-classifier-resnet18 --task_name=resnet18-20251205 --device=0,1,2,3 


accelerate launch  fastai/train.py --distributed  --data_path /mnt/ssd/ai-classify-all-data --img_size=224 --batch_size=128 --arch=convnext_atto --epochs=1000 --project_name=ai-classifier-convnext_atto --task_name=convnext_atto-20251127

accelerate launch fastai/train.py --distributed  --data_path /mnt/ssd/ai-classify-all-data --img_size=224 --batch_size=128 --arch=convnext_atto --epochs=1000 --project_name=ai-classifier-convnext_atto --task_name=convnext_atto-20251127


python fastai/train.py --data_path=/mnt/ssd/ai-classify-data --model_path=ai-classifier --img_size=224 --batch_size=128 --epochs=500  --project_name=ai-classifier task_name=resnet18

python fastai/export_onnx.py --model /mnt/ssd/ai-classify-data/models/ai-classifier.pth --data_path=/mnt/ssd/ai-classify-data --output_path=ai-classifier.onnx 

python fastai/export_onnx.py --model=/mnt/ssd/ai-classify-data/models/best.pth --output_path=ai-classifier.onnx.resnet18-v20251106 --data_path=/mnt/ssd/ai-classify-data/ --img_size=224 --arch=resnet18

python fastai/predict.py --model_path=/mnt/ssd/ai-classify-data/models/ai-classifier.onnx  --data_path=/mnt/ssd/ai-classify-data/val

python fastai/train.py --data_path=/mnt/ssd/ai-classify-data --model_path=ai-classifier-b1 --img_size=224 --batch_size=32 --arch=efficientnet_b1 --epochs=500 --project_name=ai-classifier-b1 --task_name=20251031

```

## yolov11 直接识别三分类

直接使用 yolov11 识别三分类的线框
```sh
wsictl view get-box --dir=/mnt/ssd/ai-three-detection-data --tars=tars-prod.conf --full --tags=AIM,AIC --labels=docs/ai-classifier/ai-three-detection.json --storeKey=ai

. ./step_mlflow.sh

python yolo/train.py --data docs/ai-classifier/ai-three-detection.yaml --model yolo11l.pt --epochs 1000 --imgsz 640 --batch 2 --project ai-three-detection-yolo --task_name three-detection.20251203 --device 0,1,2

python yolo/export_onnx.py --model_path runs/ai-detection-three-yolo/three-detection.20251203/weights/best.pt --imgsz=640 --output=runs/ai-three-detection-yolo/detection-three.20251203/weights/best.onnx

```


## 单个分类模型

```sh
wsictl view get-box-image --dir=/mnt/ssd/ai-single-data --tars=tars-prod.conf --tags=AIM,AIC --labels=docs/ai-classifier/ai-classify-1.json --storeKey=ai

. ./step_mlflow.sh

yolo classify train data=/mnt/ssd/ai-single-data model=yolo11n-cls.pt imgsz=224 batch=1024 epochs=1000 project=ai-classifier-single name=yolov11.20251205  patience=50 cos_lr=True lr0=0.005 lrf=0.1 optimizer=AdamW device=4,5,6,7

python yolo/export_onnx.py --model_path ai-classifier-single/yolov11.20251205/weights/best.pt --imgsz=224 --output=ai-classifier-single/yolov11.20251205/weights/best.onnx

```

## 成团分类模型

```sh
wsictl view get-box-image --dir=/mnt/ssd/ai-cluster-data --tars=tars-prod.conf --tags=AIM,AIC --labels=docs/ai-classifier/ai-classify-1.json --storeKey=ai

. ./step_mlflow.sh


yolo classify train data=/mnt/ssd/ai-cluster-data model=yolo11n-cls.pt imgsz=224 batch=1024 epochs=1000 project=ai-classifier-cluster name=yolov11.20251206  patience=50 cos_lr=True lr0=0.005 lrf=0.1 optimizer=AdamW

python yolo/export_onnx.py --model_path ai-classifier-cluster/yolov11.20251205/weights/best.pt --imgsz=224 --output=ai-classifier-cluster/yolov11.20251205/weights/best.onnx

```


## 微生物分类模型

```sh
wsictl view get-box-image --dir=/mnt/ssd/ai-micro-data --tars=tars-prod.conf --tags=AIM,AIC --labels=docs/ai-classifier/ai-classify-2.json --storeKey=ai

. ./step_mlflow.sh


yolo classify train data=/mnt/ssd/ai-micro-data model=yolo11n-cls.pt imgsz=224 batch=256 epochs=1000 project=ai-classifier-micro name=yolov11.20251206  patience=50 cos_lr=True lr0=0.05 lrf=0.1 optimizer=AdamW device=4,5

python yolo/export_onnx.py --model_path ai-classifier-micro/yolov11.20251205/weights/best.pt --imgsz=224 --output=ai-classifier-micro/yolov11.20251205/weights/best.onnx

```

## hugging 测试

python hugging/train.py --data_path=/mnt/ssd/ai-classify-all-data --model_path=ai-classifier-resnet18 --img_size=224 --batch_size=32 --arch=resnet18 --epochs=500 --project_name=ai-classifier-resnet18 --task_name=hugging-test --device=7 --train_size=1024 --val_size=256
