import os
import argparse
import yaml
import time
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics
from torch.cuda.amp import autocast, GradScaler


from model import SwinClassifier
from dataset import (
    Sen3ClassesDataset, 
    prepare_sen3classes_splits, 
    get_sen3classes_transforms
)
from utils import print_trainable_parameters_summary

try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# 0. Seeding & Setup
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"--- Global Seed set to {seed} ---")

def get_head_parameters(model):
    if hasattr(model, 'head'):
        return model.head.parameters()
    else:
        raise AttributeError("Could not find classification head (checked .head)")

def run_training(config_path):
    set_seed(42)

    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strategy = config['fine_tune_strategy']
    
    run_name = f"{config['run_name']}_{strategy}"
    output_dir = os.path.join(config['output_dir'], run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, 'training_log.csv')
    model_save_path = os.path.join(output_dir, 'final_model.pth')
    results_file = os.path.join(output_dir, 'results.txt')

    print(f"--- Starting Sen3Classes Run: {run_name} on {DEVICE} ---")

    # 2. Strategy & Hyperparameters
    strategy_cfg = config[strategy]
    BATCH_SIZE = strategy_cfg['batch_size']
    NUM_EPOCHS = strategy_cfg['epochs']
    
    # 3. Data Loading 
    is_binary = config['data'].get('binary', False)
    train_paths, val_paths, train_labels, val_labels, class_names = prepare_sen3classes_splits(
        config['data']['root_dir'], 
        train_size=config['data']['train_split'],
        binary=is_binary
    )
    
    train_tf, val_tf = get_sen3classes_transforms(config['data']['input_size'])
    
    train_dataset = Sen3ClassesDataset(train_paths, train_labels, transform=train_tf)
    val_dataset = Sen3ClassesDataset(val_paths, val_labels, transform=val_tf)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=config['data']['num_workers'], 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=config['data']['num_workers'], 
        pin_memory=True
    )

    # 4. Model Instantiation
    model_kwargs = {
        'input_channels': config['model']['input_channels'],
        'num_classes': config['data']['num_classes'],
        'pretrained_path': config['model'].get('pretrained_path')
    }
    model = SwinClassifier(**model_kwargs).to(DEVICE)

    # --- Apply Fine-Tuning Strategy ---
    params_to_optimize = []
    if strategy in ['full', 'partial_ft']:
        if strategy == 'partial_ft':
            for name, param in model.named_parameters():
                param.requires_grad = True 
                for prefix in strategy_cfg['layers_to_freeze']:
                    if name.startswith(prefix):
                        param.requires_grad = False
                        break
        else:
            for param in model.parameters(): param.requires_grad = True

        head_ptr = get_head_parameters(model)
        head_params_ids = list(map(id, head_ptr))
        backbone_params = filter(lambda p: id(p) not in head_params_ids and p.requires_grad, model.parameters())
        head_params = get_head_parameters(model)

        params_to_optimize = [
            {'params': backbone_params, 'lr': float(strategy_cfg['backbone_lr'])},
            {'params': head_params,     'lr': float(strategy_cfg['head_lr'])}
        ]

    elif strategy == 'frozen':
        for param in model.parameters(): param.requires_grad = False
        for param in get_head_parameters(model): param.requires_grad = True
        params_to_optimize = [{'params': get_head_parameters(model), 'lr': float(strategy_cfg['head_lr'])}]

    elif strategy == 'lora':
        for param in model.parameters(): param.requires_grad = False
        peft_config = LoraConfig(
            r=strategy_cfg['r'], lora_alpha=strategy_cfg['lora_alpha'],
            target_modules=strategy_cfg['target_modules'], lora_dropout=strategy_cfg['lora_dropout'], bias=strategy_cfg['bias']
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if 'head' in name: param.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        params_to_optimize = [{'params': trainable_params, 'lr': float(strategy_cfg['learning_rate'])}]

    print_trainable_parameters_summary(model)

    # 5. Optimizer & Metrics
    optimizer = optim.AdamW(params_to_optimize, betas=(0.9, 0.999), weight_decay=float(strategy_cfg['weight_decay']))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    warmup_epochs = config['training']['warmup_epochs']
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    num_classes = config['data']['num_classes']
    train_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(DEVICE)
    val_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(DEVICE)
    val_f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(DEVICE)
    val_class_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average=None).to(DEVICE)

    use_amp = True
    amp_dtype = torch.bfloat16 if config['training']['amp_dtype'] == 'bfloat16' else torch.float16
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'head_lr', 'backbone_lr', 'time'])

    # 6. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    total_start_time = time.time()
    epoch_times = []

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        train_acc_metric.reset()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            with autocast(dtype=amp_dtype, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            train_acc_metric.update(outputs, labels)

        scheduler.step()
        epoch_dur = time.time() - epoch_start
        epoch_times.append(epoch_dur)
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = train_acc_metric.compute().item()
        
        current_head_lr = optimizer.param_groups[-1]['lr']
        current_backbone_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 1 else 0.0

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {epoch_dur:.1f}s")
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, epoch_loss, epoch_acc, current_head_lr, current_backbone_lr, epoch_dur])

    total_time = time.time() - total_start_time
    print("--- Training Complete ---")

    # 7. Final Evaluation
    print("\n--- Starting Final Evaluation on Validation Set ---")
    val_start = time.time()
    model.eval()
    val_acc_metric.reset(); val_f1_metric.reset(); val_class_acc_metric.reset()

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            with autocast(dtype=amp_dtype, enabled=use_amp):
                outputs = model(images)
            val_acc_metric.update(outputs, labels)
            val_f1_metric.update(outputs, labels)
            val_class_acc_metric.update(outputs, labels)

    final_acc = val_acc_metric.compute().item()
    final_f1 = val_f1_metric.compute().item()
    per_class_acc = val_class_acc_metric.compute()
    val_dur = time.time() - val_start

    torch.save(model.state_dict(), model_save_path)

    with open(results_file, 'w') as f:
        f.write(f"--- Results: {run_name} ---\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Total Time: {total_time/60:.2f} mins\n")
        f.write(f"Final Validation Accuracy: {final_acc:.6f}\n")
        f.write(f"Final Validation F1 (Macro): {final_f1:.6f}\n\n")
        f.write("--- Per-Class Accuracy ---\n")
        
        for i, acc in enumerate(per_class_acc):
            # Using the dynamic class_names returned from prepare_sen3classes_splits
            name = class_names[i] if i < len(class_names) else f"Class {i}"
            f.write(f"{name: <25}: {acc.item():.4f}\n")

    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_training(args.config)