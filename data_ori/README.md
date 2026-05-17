# 原始数据集下载

将数据集下载后解压到此目录，目录名须与配置文件匹配。

| 数据集 | 下载地址 | 存放目录 |
|--------|----------|----------|
| iBUG Fabrics | [Kaggle](https://www.kaggle.com/datasets/orchit/the-fabrics-dataset-by-ibug) | `data_ori/ibug/` |
| WISUDAA | [Roboflow](https://universe.roboflow.com/ta-rgrwi/wisudaa) | `data_ori/wisudaa/` |
| Fabric YOLO26 | [Roboflow](https://universe.roboflow.com/sai-amar-cv4kg/fabric-qxgmo/dataset/1) | `data_ori/fabric_yolo26/` |
| Deep Learning Testing | [Roboflow](https://universe.roboflow.com/group-12-mcgzh/deep-learning-testing/dataset/1) | `data_ori/deep_learning_testing/` |

## 目录结构要求

### Folder 格式 (ibug)

```
data_ori/ibug/
├── Corduroy/
│   ├── sample_001/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── ...
├── Cotton/
└── ...
```

### Roboflow Multiclass 格式 (wisudaa, deep_learning_testing)

```
data_ori/wisudaa/
├── train/
│   ├── _classes.csv
│   ├── image_001.jpg
│   └── ...
├── valid/
│   ├── _classes.csv
│   ├── image_002.jpg
│   └── ...
```

### YOLO Detection 格式 (fabric_yolo26)

```
data_ori/fabric_yolo26/
├── data.yaml
├── train/
│   ├── images/
│   │   └── *.jpg
│   └── labels/
│       └── *.txt
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
