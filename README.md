# Airport_Classification

This repository consists of a curated Sentinel-1 dataset for recognizing Airport POIs. The dataset is divided into 3 classes: **airports**, **bus-stands**, and **railway stations**. The pre-trained checkpoint and scripts for fine-tuning the **FLASH-SAR** model on this dataset are provided.

---

## 🚀 Getting Started

### Step 1: Clone the Repository

Since this repository uses **Git LFS** to manage large model weights and datasets, ensure `git-lfs` is installed on your system before cloning.

```bash
# Install Git LFS (if not already installed)
sudo apt-get install git-lfs
git lfs install

# Clone the repository
git clone https://github.com/wannabe-yoda/Airport_Classification.git
cd Airport_Classification

# Pull the large files (weights and dataset)
git lfs pull
```

### Step 2: Install Dependencies

This project requires Python 3.8+ and several deep learning libraries. It is recommended to use a virtual environment or a Conda environment.

```bash
# Core Deep Learning Framework
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Model Architectures & Parameter-Efficient Fine-Tuning
pip install transformers peft

# Data Processing & Utilities
pip install numpy tifffile scikit-learn pyyaml tqdm

# Evaluation Metrics
pip install torchmetrics
```

## 🏋️ Training & Benchmarking

Once your environment is set up, you can begin training. The repository provides two primary configurations: Multiclass (3-class) and Binary (2-class).

### Step 3: Run the Training Script

To launch a training run, use the --config flag followed by the path to your desired configuration file.

For 3-Class Training (Airports vs. Bus Stands vs. Railway Stations):
```bash
python train.py --config config_3classes.yaml
```
For 2-Class Training (Airports vs. Non-Airport Hubs):

```bash
python train.py --config config_2classes.yaml
```

Configuration Overview

The .yaml files manage the data splits, model parameters, and training strategies. By default, both are set to Linear Probing (frozen).
| Feature | `config_3classes.yaml` | `config_2classes.yaml` |
|---|---|---|
| Num Classes | 3 | 2 |
| Logic | Individual POI Recognition | Airport vs. Transportation Hubs |
| Default Strategy | Frozen Backbone | Frozen Backbone |
| Input Channels | 2 (VV, VH) | 2 (VV, VH) |

Modify the fine_tune_strategy key in the config files to change the training behavior:

    frozen (Default): Linear probing. Only the classification head is trained.
    full: Full fine-tuning with differential learning rates.
    lora: Low-Rank Adaptation using the peft library.
    partial_ft: Freezes early layers to preserve low-level SAR features.


## 📊 Results & Logging

Results are saved to the experiments/ directory:

    training_log.csv — Per-epoch Loss and Accuracy.
    results.txt — Final report with Global Accuracy, F1-Macro, and Per-Class Accuracy.
    final_model.pth — The saved weights of your fine-tuned model.

## 📜 License & Citation

This repository is for research purposes. If you use the dataset or the FLASH-SAR checkpoint, please cite the corresponding work.

BibTeX:
```bash

@inproceedings{prakhya2026flash,
  title={FLASH-SAR: Fast Learning Self-supervised Hierarchical Architecture for SAR},
  author={Prakhya, Sai Shruti and Kumar, Uttam},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1382--1392},
  year={2026}
}
```

