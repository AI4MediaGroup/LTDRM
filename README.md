# Exploring coordinated motion patterns of facial landmarks for deepfake video detection

## 📖 Citation

> **Exploring coordinated motion patterns of facial landmarks for deepfake video detection**  
> Zhang Y, Niu R, Zhang X, et al. *Applied Soft Computing*, 2025.  
> 🔗 [https://www.sciencedirect.com/science/article/pii/S1568494625002856](https://www.sciencedirect.com/science/article/pii/S1568494625002856)

---

## ⚙️ Requirements

Install the following packages:

```
einops fvcore timm
torch==1.13.0 torchvision==0.14.0
```

## 🗂️ Dataset

1. Frame Extraction

    Extract video frames and crop faces using MTCNN

2. Landmark Detection & Calibration

    Detect landmark coordinates using OpenFace, then calibrate them with the LRNet calibration module

3. Prepare a file listing video samples with the following format per line:
```
<image_folder> <landmark_file> <start_frame> <end_frame> <label>
```
Example:
```
ff++_images/Deepfakes001_870 ff++_landmarks/Deepfakes001_870.txt 1 460 1
```

4. Organize your dataset in the following structure:
```
dataset/
├── TALL_ff++_list_images_landmarks/
│   ├── ff++_train_fold.txt
│   ├── ff++_val_fold.txt
│   ├── ff++_test_fold.txt
│   └── ...
├── ff++_images/
│   ├── Deepfakes000_003/
│   │    ├── 00001.jpg
│   │    ├── 00002.jpg
│   │    └── ...
│   └── ...
└── ff++_landmarks/
    ├── Deepfakes000_003.txt
    ├── Deepfakes001_870.txt
    └── ...
```

## Train

```
python main.py
```

## Evaluation

```
python test.py
```

## 