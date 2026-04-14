import torch
import torch.nn as nn
from transformers import SwinConfig, SwinModel

class SwinClassifier(nn.Module):
    def __init__(self, input_channels=2, num_classes=3, pretrained_path=None):
        super().__init__()

        # 1. Encoder: Swin Transformer
        # Using Swin-Tiny config as the base
        self.config = SwinConfig(
            image_size=224,
            patch_size=4,
            window_size=7,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            num_channels=input_channels, 
            output_hidden_states=False   
        )
        self.encoder = SwinModel(self.config)

        # 2. Classification Head
        # Swin-Tiny output dim is 8 * embed_dim = 768
        self.head = nn.Linear(self.config.embed_dim * 8, num_classes)

        # 3. Load Weights if a path is provided
        if pretrained_path:
            self.load_custom_weights(pretrained_path)

    def forward(self, x):
        # x shape: [Batch, input_channels, 224, 224]
        
        # Encoder Output
        outputs = self.encoder(x)
        last_hidden_state = outputs.last_hidden_state 
        # Shape: [Batch, Num_Patches, Hidden_Dim] -> [B, 49, 768]

        # Global Average Pooling (Pool over the patches)
        pooled_output = last_hidden_state.mean(dim=1) 
        # Shape: [B, 768]

        # Classification
        logits = self.head(pooled_output)
        return logits

    def load_custom_weights(self, path):
        """
        Smart loader that handles specific prefixes found in checkpoint ('backbone.').
        """
        print(f"Loading weights from: {path}")
        state_dict = torch.load(path, map_location='cpu')
        
        # If the file contains the whole 'model' object or a dict with 'state_dict' key
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            # Handle the 'backbone.' prefix
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '')
                new_state_dict[new_key] = v
            
            # support for 'encoder.' prefix
            elif k.startswith('encoder.'):
                new_key = k.replace('encoder.', '')
                new_state_dict[new_key] = v
            
            # Direct match
            elif k in self.encoder.state_dict():
                new_state_dict[k] = v
                
        # Load into the encoder only
        
        missing, unexpected = self.encoder.load_state_dict(new_state_dict, strict=False)
        
        print(f"Weights loaded. Missing keys: {len(missing)}.")
        
        # Verifying
        if len(missing) > 10:
            print("!!! WARNING: High number of missing keys. Check prefixes again! !!!")
            print(f"Example missing keys: {missing[:5]}")
        else:
            print("Success: Backbone weights loaded correctly.")