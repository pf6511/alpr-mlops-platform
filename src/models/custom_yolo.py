import torch
import torch.nn as nn
import os

class ConvBlock(nn.Module):
    """Conv2D + BatchNorm + LeakyReLU"""
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SimpleYOLO(nn.Module):
    """
    Simplified YOLO detector inspired by the notebook.
    Input: (batch, 3, 416, 416)
    Output: (batch, 13, 13, 6) = [obj, x, y, w, h, class]
    """
    def __init__(self, grid_size=13, num_classes=1):
        super().__init__()
        self.grid_size = grid_size
        self.output_ch = 5 + num_classes
        
        # Backbone: 416->208->104->52->26->13
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 64, 1, 1, 0),
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )
        
        # Head
        self.head = nn.Sequential(
            ConvBlock(512, 1024, 3, 1, 1),
            nn.Dropout(0.5),
            ConvBlock(1024, 512, 1, 1, 0),
            nn.Dropout(0.5),
            ConvBlock(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, self.output_ch, 1, 1, 0)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)
        
        # Activations
        obj = torch.sigmoid(x[..., 0:1])
        xy = torch.sigmoid(x[..., 1:3])
        wh = x[..., 3:5]
        cls = torch.sigmoid(x[..., 5:])
        
        return torch.cat([obj, xy, wh, cls], dim=-1)

def load_custom_model(model_path, device='cpu'):
    """Utility function to load state_dict into SimpleYOLO."""
    model = SimpleYOLO(grid_size=13, num_classes=1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
