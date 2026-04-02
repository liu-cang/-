# SFUOD: Source-Free Unknown Object Detection (ICCV 2025)
*Keon-Hee Park, Seun-An Choe, Gyeong-Moon Park*

Official Pytorch implementation of [`SFUOD: Source-Free Unknown Object Detection, ICCV 2025`](https://arxiv.org/pdf/2507.17373).


## 1. Installation

### 1.1 Requirements

- Linux, CUDA >= 11.1, GCC >= 8.4

- Python >= 3.8

- torch >= 1.10.1, torchvision >= 0.11.2

- Other requirements

  ```bash
  pip install -r requirements.txt
  ```

### 1.2 Compiling Deformable DETR CUDA operators

```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## 2. Dataset Preparation

Weather Adaptation:

 Cityscapes (source domain) → FoggyCityscapes with foggy level 0.02 (target domain).

You can download the raw data from the official websites: [Cityscapes](https://www.cityscapes-dataset.com/downloads/),  [FoggyCityscapes](https://www.cityscapes-dataset.com/downloads/).  The annotations, converted into COCO format, can download from [here](https://drive.google.com/file/d/1qa6rXaVqWuef_YA5q4oUJH2v385mDtpp/view?usp=sharing).
The datasets and annotations are organized as:

```bash
[data_root]
└─ cityscapes
	└─ annotations
		└─ cityscapes_train_cocostyle.json
		└─ cityscapes_val_cocostyle.json
	└─ leftImg8bit
		└─ train
		└─ val
└─ foggy_cityscapes
	└─ annotations
		└─ foggy_cityscapes_train_cocostyle.json
		└─ foggy_cityscapes_val_cocostyle.json
	└─ leftImg8bit_foggy
		└─ train
		└─ val
```

## 3. Training
first edit the files in `configs/def-detr-base/city2foggy/` to specify your own `DATA_ROOT` and `OUTPUT_DIR`, then run:

### 3.1 Source Training
```bash
sh configs/def-detr-base/city2foggy/source_only_sfuod.sh
```
### 3.2 Target Training
Mean-Teacher framework
```bash
sh configs/def-detr-base/city2foggy/teaching_standard_sfuod.sh
```
DRU (ECCV 2024)
```bash
sh configs/def-detr-base/city2foggy/teaching_mask_sfuod.sh
```
Target Training (Oracle)
```bash
sh configs/def-detr-base/city2foggy/target_only_sfuod.sh
```


## 4. Acknowledgement
This implementation is built upon [DRU](https://github.com/lbktrinh/DRU), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), and [MIC](https://github.com/lhoyer/MIC). We sincerely appreciate their contributions.
