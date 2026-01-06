## 项目说明

根据 AI 分析的结果(细胞数量),训练模型,对数据的阴阳性做分类.

## 数据准备

首先在标注平台上,新建 AI 分类任务, 对平台上的玻片做分析，分析完毕后，可以通过 wsictl 工具将每张玻片的细胞数量，平均置信度拉下来，形成一张 csv 文件

```sh
wsictl view get-image-class --tars=tars-prod.conf --dir=/mnt/ssd/ai-image-class --tags=pgTest --storeKey=img --modelDetection=ai-detection.onnx.yolov11-v20251126 --modelClassification=ai-classifier.onnx.convnextatto-v20251217
```

## 训练过程

```sh

conda activate py3.12

python fastai/train_tabular.py  --target_col label --use_all_features --hidden_dims 128,64,128 --epochs 300 
    --batch_size 64 --lr 0.001 --dropout 0.5 --project_name positive_nagative_classifer --data /mnt/ssd/ai-image-class/image_class_stats.csv --task_name pn-20251229

python fastai/predict_tabular.py --use_all_features --data /mnt/ssd/ai-image-class/image_class_stats.csv   --model best.pt

bin/inferclient --command=infer --model=/home/heer/ruanshudong/box-classification/runs/positive_nagative_classifer/pn-20251231/best.onnx --csv=/mnt/ssd/ai-image-class/image_class_stats.csv

```