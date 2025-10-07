# Evaluating Pretrained General-Purpose Audio Representations for Music Genre Classification  

## Project Overview  
This project focuses on classifying music genres based on audio features extracted from different datsets like GTZAN and FMA. The approach involves using **BYOL (Bootstrap Your Own Latent) for Audio** to extract embeddings and training classifiers on these features.  

---

# Dataset Information  

## 1. GTZAN Dataset  
- Contains **GTZAN audio files**.  
- Used for training and evaluating genre classification models.  

## 2. FMA-Small Dataset  
- Includes a **subset** of the **FMA (Free Music Archive) dataset**.  
- Contains:  
  - Audio files  
  - `genres.csv`, `tracks.csv`  
  - `train-split.csv`, `test-split.csv` (created manually)  

---


# Experiment Results

## Dataset: GTZAN  
**Feature Extraction using: BYOLA**

| Experiment | Objective                          | Hidden Layers & Nodes          | Activation | Dropout | BatchNorm | Optimizer | Noise Augmentation                                      | Weight Clipping   | Batch Size | Learning Rate | Max Epochs | Patience | Avg Best Val Acc | Avg Train Loss | Avg Val Loss | Avg Test Acc |
|------------|-----------------------------------|--------------------------------|------------|---------|-----------|-----------|-------------------------------------------------------|------------------|-----------|-------------|-----------|---------|----------------|--------------|--------------|-------------|
| Exp 1.1   | Baseline Model                    | 1 Hidden (64)                  | ReLU       | None    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 83.35          | 0.146        | 0.736        | 76         |
| Exp 2.1   | Increase depth                    | 2 Hidden (64,32)               | ReLU       | None    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 81.23          | 0.373        | 0.769        | 77           |
| Exp 2.2   | Increase depth                    | 3 Hidden (128,64,32)           | ReLU       | None    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 81.60          | 0.458        | 0.871        | 78.5         |
| Exp 2.3   | Increase depth                    | 4 Hidden (256,128,64,32)       | ReLU       | None    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 81.10          | 0.566        | 1.048        | 76.5         |
| Exp 3.1   | Increase nodes by 1.5x            | 4 Hidden (256,128,64,32)       | ReLU       | None    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 81.60          | 0.479        | 0.942        | 74.5         |
| Exp 3.2   | Increase nodes by 2x                  | 4 Hidden        | ReLU       | None    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 81.10          | 0.566        | 1.048        | 76.5         |
| Exp 3.3   | decrease nodes by 0.5x            | 4 Hidden        | ReLU       | None    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 79.72          | 0.694       | 1.133      | 78.5         |
| Exp 4.1   | Add dropout(0.2)                    | 4 Hidden        | ReLU       | 0.2    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 82.35        | 0.756        | 1.033       | 77.5         |
| Exp 4.2   | Add dropout(0.4)            | 4 Hidden        | ReLU       | 0.4    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 82.60          | 0.99       | 1.081      | 79         |
| Exp 4.3   | Add dropout(0.5)                    | 4 Hidden        | ReLU       | 0.5    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 80.72          | 1.203       | 1.187        | 79        |
| Exp 5.1   | Replace ReLU with LeakyReLU       | 4 Hidden (256,128,64,32)       | LeakyReLU  | None    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 82.48          | 0.439        | 0.885        | 76.5         |
| Exp 5.2   | Replace ReLU with ELU             | 4 Hidden (256,128,64,32)       | ELU        | None    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 81.85          | 0.535        | 1.019        | 78           |
| Exp 5.3   | Replace ReLU with Swish           | 4 Hidden (256,128,64,32)       | Swish      | None    | None      | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 85.48          | 0.371        | 0.886        | 78        |
| Exp 6.1   | Add BatchNorm                     | 4 Hidden (256,128,64,32)       | ReLU       | None    | Yes       | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 85.23          | 0.858        | 1.034        | 81           |
| Exp 6.2   | BatchNorm + Dropout (0.3)         | 4 Hidden (256,128,64,32)       | ReLU       | 0.3     | Yes       | Adam      | No                                                    | No               | 64        | 0.0005       | 50        | 7       | 85.23          | 0.858        | 1.034        | 81           |
| Exp 7.1   | BatchNorm + Dropout               | 4 Hidden (256,128,64,32)       | ReLU       | 0.3     | Yes       | Adam      | Yes (once at the beginning) noise factor = 0.1                          | No               | 64        | 0.0005       | 50        | 7       | 97.87          | 0.50        | 0.43        | 81.5           |
| Exp 7.2   | BatchNorm + Dropout               | 4 Hidden (256,128,64,32)       | ReLU       | 0.3     | Yes       | Adam      | Yes (at the beginning of each epoch) noise factor = 0.1                           | No               | 64        | 0.0005       | 50        | 7       | 72.83          | 1.77        | 1.87        | 81.0           |
| Exp 7.3   | BatchNorm + Dropout               | 4 Hidden (256,128,64,32)       | ReLU       | 0.3     | Yes       | Adam      | No                           | Yes(clip_value=0.5)               | 64        | 0.0005       | 50        | 7       | 43.35         | 2.06        | 2.22        | 81.0          |
| Exp 7.4   | BatchNorm + Dropout               | 4 Hidden (256,128,64,32)       | ReLU       | 0.3     | Yes       | Adam      | No                           | Yes (once at the beginning) noise factor = 0.1                | 64        | 0.0005       | 35        | 5       | 97.68          | 0.651        | 0.547        | 81.5           |


## Dataset: FMA (small)  
**Feature Extraction using: BYOLA**

| Experiment | Objective             | Hidden Layers & Nodes          | Activation | Dropout | BatchNorm | Optimizer | Noise Augmentation | Weight Clipping | Batch Size | Learning Rate | Max Epochs | Patience | Avg Best Val Acc | Avg Train Loss | Avg Val Loss | Avg Test Acc |
|------------|----------------------|--------------------------------|------------|---------|-----------|-----------|-------------------|----------------|-----------|-------------|-----------|---------|----------------|--------------|--------------|-------------|
| Exp 1.1   | BatchNorm + Dropout  | 4 Hidden (256,128,64,32)       | ReLU       | 0.3     | Yes       | Adam      | Yes (once at the beginning) | No         | 64        | 0.0005       | 35        | 5       | 95.38          | 0.584        | 0.803        | 64.31       |

## Dataset: GTZAN  
**Feature Extraction using: VGGish**

| Experiment | Objective             | Hidden Layers & Nodes          | Activation | Dropout | BatchNorm | Optimizer | Noise Augmentation | Weight Clipping | Batch Size | Learning Rate | Max Epochs | Patience | Avg Best Val Acc | Avg Train Loss | Avg Val Loss | Avg Test Acc |
|------------|----------------------|--------------------------------|------------|---------|-----------|-----------|-------------------|----------------|-----------|-------------|-----------|---------|----------------|--------------|--------------|-------------|
| Exp 1.1   | BatchNorm + Dropout  | 4 Hidden (256,128,64,32)       | ReLU       | 0.3     | Yes       | Adam      | Yes (once at the beginning) | No         | 64        | 0.0005       | 35        | 5       | 98.31          | 0.585        | 0.487        | 79.5        |

## Dataset: FMA (small)  
**Feature Extraction using: VGGish**

| Experiment | Objective             | Hidden Layers & Nodes          | Activation | Dropout | BatchNorm | Optimizer | Noise Augmentation | Weight Clipping | Batch Size | Learning Rate | Max Epochs | Patience | Avg Best Val Acc | Avg Train Loss | Avg Val Loss | Avg Test Acc |
|------------|----------------------|--------------------------------|------------|---------|-----------|-----------|-------------------|----------------|-----------|-------------|-----------|---------|----------------|--------------|--------------|-------------|
| Exp 1.1   | BatchNorm + Dropout  | 4 Hidden (256,128,64,32)       | ReLU       | 0.3     | Yes       | Adam      | Yes (once at the beginning) | No         | 64        | 0.0005       | 35        | 5       | 98.22          | 0.386        | 0.900        | 58.04       |

---
## Dataset: GTZAN  
**Feature Extraction using: PANNS**

| Experiment | Objective             | Hidden Layers & Nodes          | Activation | Dropout | BatchNorm | Optimizer | Noise Augmentation | Weight Clipping | Batch Size | Learning Rate | Max Epochs | Patience | Avg Best Val Acc | Avg Train Loss | Avg Val Loss | Avg Test Acc |
|------------|----------------------|--------------------------------|------------|---------|-----------|-----------|-------------------|----------------|-----------|-------------|-----------|---------|----------------|--------------|--------------|-------------|
| Exp 1.1   | BatchNorm + Dropout  | 4 Hidden (256,128,64,32)       | ReLU       | 0.3     | Yes       | Adam      | Yes (once at the beginning) | No         | 64        | 0.0005       | 35        | 5       | 74.62          | 0.938        | 1.212        | 77        |

