# yueli 面料数据集

### 目录结构

```
data_ori/yueli/
├── cotton/       # 棉布 (48 张)
├── other/        # 其他 (8 张)
└── polyester/    # 聚酯纤维 (55 张)
```

### 格式说明

- 格式: 文件夹结构（类别名作为目录名，目录内直接存放图片）
- 图片格式: `.jpg`
- 总图片: ~111 张
- 类别: 3 类（cotton / other / polyester）

### 预处理

```bash
python main.py prepare yueli
```

预处理会将原始图片按类别复制到 `data/dataset_name/train/` 和 `data/dataset_name/val/` 目录下，并按全局 8 类映射归并。
