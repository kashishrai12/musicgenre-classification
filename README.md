# Music Genre Classification Using Audio Features  

## Project Overview  
This project focuses on classifying music genres based on audio features extracted from different datsets like GTZAN and FMA. The approach involves using **BYOL (Bootstrap Your Own Latent) for Audio** to extract embeddings and training classifiers on these features.  

---

## Dataset Information  

### 1. GTZAN Dataset  
- Contains **GTZAN audio files**.  
- Used for training and evaluating genre classification models.  

### 2. FMA-Small Dataset  
- Includes a **subset** of the **FMA (Free Music Archive) dataset**.  
- Contains:  
  - Audio files  
  - `genres.csv`, `tracks.csv`  
  - `train-split.csv`, `test-split.csv` (created manually)  

**Note:** These datasets are **not included in the repository** due to their large size.  

---

## Folder Structure
```
│── dataset/ # GTZAN dataset (Not included in repo)
│── dataset2/ # FMA-Small dataset (Not included in repo)
│── preprocessed/ # Audio embeddings from GTZAN using BYOLA
│── preprocessed-fma/ # Audio embeddings from FMA-Small using BYOLA
│── scripts/ # All scripts for preprocessing & training
│ ├── preprocess.py # Preprocess GTZAN dataset
│ ├── save-features.py # Extract BYOLA embeddings from GTZAN
│ ├── load-metadata.py # Loads metadata from FMA (tracks.csv, genres.csv)
│ ├── prepare-dataset.py # Filters FMA tracks.csv for "small" subset
│ ├── create-train-test-split.py # Creates train & test splits for FMA-Small
│ ├── preprocess-fma-small.py # Preprocesses FMA-Small dataset
│ ├── extract-embeddings-fma.py # Extracts BYOLA embeddings for FMA-Small
│ ├── map-embeddings-to-labels.py # Maps BYOLA embeddings to genre labels (FMA)
│ ├── exp1 to 7_1.py # Experiments 1 to 7_1 (GTZAN)
│ ├── exp-noise-augmentation-epoch.py # Noise augmentation per epoch (GTZAN)
│ ├── exp-noise-augmentation-once.py # Noise augmentation once (GTZAN)
│ ├── exp-weight-clipping.py # Weight clipping experiment (GTZAN)
│ ├── exp-noise-augmentation-once-fma.py # Best model training on FMA-Small
│── plots/ # Visualization results
│── .gitignore # Ignore large files & dataset folders
│── .gitattributes # Git settings for large files
│── README.md # Project documentation
```

---

## Key Scripts and Their Functions  

### GTZAN Processing  
- `preprocess.py`: Preprocesses the **GTZAN** dataset.  
- `save-features.py`: Extracts **BYOL embeddings** from GTZAN audio.  

### GTZAN Training Experiments  
- `exp1-1.py`: First experiment on GTZAN.  
- `exp-noise-augmentation-epoch.py`: Noise augmentation per epoch.  
- `exp-noise-augmentation-once.py`: Noise augmentation applied once.  
- `exp-weight-clipping.py`: Weight clipping experiment.  

### FMA-Small Processing  
- `load-metadata.py`: Loads **FMA-Small metadata** (`tracks.csv`, `genres.csv`).  
- `prepare-dataset.py`: Filters `tracks.csv` to include only "small" subset.  
- `create-train-test-split.py`: Creates **train-test splits**.  
- `preprocess-fma-small.py`: Preprocesses **FMA-Small dataset**.  
- `extract-embeddings-fma.py`: Extracts **BYOL embeddings** for train & test sets.  
- `map-embeddings-to-labels.py`: Maps **BYOL embeddings** to **genre labels**.  

### FMA-Small Training Experiment  
- `exp-noise-augmentation-once-fma.py`: Best-performing classifier training on **FMA-Small**.  

---

