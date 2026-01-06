## 项目说明

DNA 多光谱识别团聚细胞 以及 团细胞分类

### 训练数据准备

- 获取全量标注视野图
```sh
wsictl view get-box --dir=/mnt/ssd/dna-cluster-one-box-data --tars=tars-prod.conf --full --tags=DNA --labels=docs/dna-cluster/dna-cluster-one.json --storeKey=ai
```

数据会生成到 `dir`, 目录下, 以供训练.

### 下载 yolov5的代码

```sh
conda create -n yolov5 python=3.10
conda activate yolov5
git clone https://github.com/ultralytics/yolov5
```

### 训练过程

```sh
conda activate yolov5
export PYTHONWARNINGS="ignore::FutureWarning"
python yolov5/train.py --img 640 --epochs 200 --data docs/dna-cluster/dna-cluster-one.yaml --weights yolov5s.pt --name dna-cluster-one-s.20250829
python yolov5/train.py --img 640 --epochs 200 --data docs/dna-cluster/dna-cluster-one.yaml --weights yolov5m.pt --name dna-cluster-one-m

```

### Detect

```sh
python yolov5/detect.py --weights yolov5/runs/train/exp14/weights/best.pt --img 640 --source /mnt/ssd/dna-box-data/images/val/V010paozhen1501/

```

### Export

```sh
python yolov5/export.py --weights yolov5/runs/train/exp14/weights/best.pt --include onnx
python yolov5/export.py --weights yolov5/runs/train/dna-cluster-one-s/weights/best.pt --include onnx
python yolov5/export.py --weights yolov5/runs/train/dna-cluster-one-tag/weights/best.pt --include onnx
```

## 小图分类

### 获取标签数据
```sh
wsictl view get-box-image --dir=/mnt/ssd/dna-classify-cluster-data --tars=tars-prod.conf --tags=DNA --labels=docs/dna-cluster/dna-classify-cluster.json --storeKey=ai
```

### 使用fastai来训练

在 git@codeup.aliyun.com:6626145b290482f52b28066d/ai-system/dna-cell-classification.git 源码目录下执行:
```sh
git clone https://codeup.aliyun.com/6626145b290482f52b28066d/ai-system/dna-cell-classification.git 
export PYTHONWARNINGS="ignore::FutureWarning"
python train.py --data_path=/mnt/ssd/dna-classify-cluster-data --model_path=dna-cluster-classifier --img_size=320 --batch_size=64

python predict.py --model /mnt/ssd/dna-classify-cluster-data/models/dna-cluster-classifier.pth --data_path=/mnt/ssd/dna-classify-cluster-data --export_onnx --onnx_path=dna-cluster-classifier.onnx

python predict.py --model_path=dna-cluster-classifier.onnx  --data_path=/mnt/ssd/dna-classify-cluster-data/val

```



### 使用yolov5做分类训练

```sh
conda activate yolov5
export PYTHONWARNINGS="ignore::FutureWarning"

python yolov5/classify/train.py --img 320 --epochs 200 --data /mnt/ssd/dna-classify-cluster-data --model yolov5s-cls.pt  --pretrained False --name dna-cluster-classify --clearml

python yolov5/export.py --weights yolov5/runs/train-cls/dna-cluster-classify11/weights/best.pt --include  onnx --img=320

python yolov5/classify/predict.py --weights yolov5/runs/train-cls/dna-cluster-classify11/weights/best.pt --img 320 --source /mnt/ssd/dna-classify-cluster-data/val/negative/20250805030
```