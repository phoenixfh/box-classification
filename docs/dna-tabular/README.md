## 项目说明

使用 dna 分析的细胞核参数来分类细胞


index 

1 拉取相应小图的带分类的数据的csv: dna {all, 18, 20}; 
2 用 fastai 下的 tabluar 模型去训练 mlp 模型，可能涉及到用那个tune.py自动调参数；  
3 评估训练出来的模型。
4 导出时确认支持batch推理？
5 onnx c++代码支持实时推理（要求支持批量）: 增加一个class MLPOnnx; 
6 分析软件改造这个推理过程


## 标注数据准备

```sh
# 1 fetch all csv: but split into train/val
./bin/wsictl view get-dna-csv --tars=../test/tars-prod.conf  --dir=./t0/all_csv --tags=SQDNAALL,SQDNA20,SQDNA18 --storageKey=dna

# 2 合并所有的 csv 文件到一个 di_all.csv
(cat /home/jd/t/git/WsiCtl/build/t0/all_csv/train/all_samples.csv && cat /home/jd/t/git/WsiCtl/build/t0/all_csv/val/all_samples.csv|grep -v dnaIndex) > di_all.csv

# 3 Shuffle and and divided to di 20/18/all, and split into train/val sets with different dnaIndex filters.

python shuffle_split_di_all_20_18.py --csv $ROOT/di_all.csv

# Outputs:
#     - $ROOT/diall/train/all_samples.csv, $ROOT/diall/val/all_samples.csv (no filter)
#     - $ROOT/di20/train/all_samples.csv, $ROOT/di20/val/all_samples.csv (dnaIndex >= 2.0)
#     - $ROOT/di18/train/all_samples.csv, $ROOT/di18/val/all_samples.csv (dnaIndex >= 1.8)


"""


```




## 使用 fastai 来训练

用 fastai 下的 tabluar 模型去训练 mlp 模型

```sh
set -e
set -x

export ditest=/home/jd/t/git/WsiCtl/build/t0/all_csv/ditest/train/all_samples.csv
export ditest_output=./fastai_output_ditest

export di20=/home/jd/t/git/WsiCtl/build/t0/all_csv/di20/train/all_samples.csv
export di20_output=./fastai_output_di20

export di18=/home/jd/t/git/WsiCtl/build/t0/all_csv/di18/train/all_samples.csv
export di18_output=./fastai_output_di18

export diall=/home/jd/t/git/WsiCtl/build/t0/all_csv/diall/train/all_samples.csv
export diall_output=./fastai_output_diall
export EPOCHS=200


PYTHONPATH=.  python fastai/train_tabular.py \
    --data $ditest \
    --target_col label \
    --use_all_features \
    --hidden_dims 512,256,128 \
    --epochs 3 \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.5 \
    --output_dir $ditest_output \
    --project_name mlp-classification \
    --task_name mlp-exp1


PYTHONPATH=.  python fastai/train_tabular.py \
    --data $di20 \
    --target_col label \
    --use_all_features \
    --hidden_dims 512,256,128 \
    --epochs $EPOCHS \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.5 \
    --output_dir $di20_output \
    --project_name mlp-classification \
    --task_name mlp-exp1



PYTHONPATH=.  python fastai/train_tabular.py \
    --data $di18 \
    --target_col label \
    --use_all_features \
    --hidden_dims 512,256,128 \
    --epochs $EPOCHS \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.5 \
    --output_dir $di18_output \
    --project_name mlp-classification \
    --task_name mlp-exp1


PYTHONPATH=.  python fastai/train_tabular.py \
    --data $diall \
    --target_col label \
    --use_all_features \
    --hidden_dims 512,256,128 \
    --epochs $EPOCHS \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.5 \
    --output_dir $diall_output \
    --project_name mlp-classification \
    --task_name mlp-exp1


```

## 评估
```
   di20:    Epoch 146/200 - Train Loss: 0.1876, Train Acc: 0.9205 | Val Loss: 0.1757, Val Acc: 0.9258
   di18:    Epoch 134/200 - Train Loss: 0.1928, Train Acc: 0.9182 | Val Loss: 0.1803, Val Acc: 0.9240
   diall:   Epoch 92/200 - Train Loss: 0.2241, Train Acc: 0.9103 | Val Loss: 0.2055, Val Acc: 0.9164
```


## 导出 onnx 示例

```sh
PYTHONPATH=. python fastai/export_onnx_tabular.py \
    --model ./fastai_output/best.pt \
    --output ./fastai_output/best.onnx \
    --onnx_verify_data ./fastai/data.csv

```


