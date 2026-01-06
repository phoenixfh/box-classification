## 项目说明

AI全标注视野图训练项目，主要包含三个阶段的训练任务：

## 项目流程
分割视野图：使用UNet网络分割视野图中的结构(UNet)

三分类任务：对分割出的结构进行单个、成团和微生物分类（0-单个，1-成团，2-微生物）(EfficientNet-B0/B1)

多分类任务：对分割后的小图进行细粒度的细胞类型分类(ConvNeXt 或 EfficientNet-B4 + Focal Loss)


## 切图数据准备

1. 获取全量标注视野图用于分割训练（不区分单个和团）
```sh
wsictl view get-box --dir=/mnt/ssd/ai-detection-data --tars=tars-prod.conf --full --tags=AIM,AIC --storeKey=ai

2. 获取标注视野图用于三分类训练
```sh
wsictl view get-box-image --dir=/mnt/ssd/ai-three --tars=tars-prod.conf --full --tags=AIM,AIC --labels=docs/ai-classifier/ai-three.json --storeKey=ai

3. 获取标注视野图用于多分类训练
```sh
wsictl view get-box-image --dir=/mnt/ssd/ai-classifier-0 --tars=tars-prod.conf --tags=AIM,AIC --labels=docs/ai-classifier/ai-classify-0.json --storeKey=ai

wsictl view get-box-image --dir=/mnt/ssd/ai-classifier-1 --tars=tars-prod.conf --tags=AIM,AIC --labels=docs/ai-classifier/ai-classify-1.json --storeKey=ai

wsictl view get-box-image --dir=/mnt/ssd/ai-classifier-2 --tars=tars-prod.conf --tags=AIM,AIC --labels=docs/ai-classifier/ai-classify-2.json --storeKey=ai


注意：多分类的训练数据需要将单个、团和微生物按照celltype分开存储，以供训练使用。



## 模型训练
1. UNet分割模型训练

```sh
python fastai/train.py --data_path=/mnt/ssd/ai-detection-data --model_path=ai-detection --img_size=320 --batch_size=32                      --arch=unet --epochs=200 --device=0,1,2,3,4,5 --project_name=ai-detection-unet --task_name=unet

2. EfficientNet三分类模型训练

```sh
python fastai/train.py --data_path=/mnt/ssd/ai-three --model_path=ai-three --img_size=320 --batch_size=32               --arch=efficientnet_b1 --epochs=200 --device=0,1,2,3,4,5 --project_name=ai-three-efficientnet --task_name=efficientnet_b1


3. ConvNeXt多分类模型训练

```sh
python fastai/train.py --data_path=/mnt/ssd/ai-classify-0 --model_path=ai-classifier-convnext --img_size=320 --batch_size=32          --arch=convnext_base --epochs=1000 --device=0,1,2,3,4,5 --project_name=ai-classifier-convnext --task_name=convnext_base


4. EfficientNet-B4多分类模型训练（使用Focal Loss）

```sh
python fastai/train.py --data_path=/mnt/ssd/ai-classify-0 --model_path=ai-classifier-b4 --img_size=320 --batch_size=32 --arch=efficientnet_b4 --epochs=200 --device=1 --project_name=ai-classifier-efficientnet --task_name=efficientnet_b4 --loss=focal

##参数说明
--data_path: 训练数据路径
--model_path: 模型保存路径
--img_size: 输入图像尺寸
--batch_size: 批次大小
--arch: 网络架构
--epochs: 训练轮数
--device: 使用的GPU设备（单卡或多卡）
--project_name: 项目名称
--task_name: 任务名称
--loss: 损失函数类型（可选）

##目录结构
/mnt/ssd/
├── ai-detection-data/          # 原始数据
├── ai-classify-0/         # 多分类训练数据
│   ├── celltype1/
│   ├── celltype2/
│   └── ...
└── ai-three-data/      # 三分类训练数据
    ├── 0-single/
    ├── 1-cluster/
    └── 2-microorganism/
