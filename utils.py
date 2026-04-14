import torch
import numpy as np


def print_trainable_parameters_summary(model):
    """
    Prints the number of trainable vs. total parameters.
    """
    trainable_params_count = 0
    total_params_count = 0
    
    for name, param in model.named_parameters():
        total_params_count += param.numel()
        if param.requires_grad:
            trainable_params_count += param.numel()
            
    print(f"\n--- Model Parameter Stats ---")
    print(f"Trainable params: {trainable_params_count:,}")
    print(f"Total params:     {total_params_count:,}")
    print(f"Trainable %:      {100 * trainable_params_count / total_params_count:.2f}%")
    print("-----------------------------\n")
