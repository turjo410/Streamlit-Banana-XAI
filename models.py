import torch
import torch.nn as nn
import torchvision.models as models
import timm
from transformers import ViTModel, DeiTModel

class CustomBananaCNN_Variety(nn.Module):
    """Custom CNN for banana variety classification - NO SOFTMAX for XAI"""
    def __init__(self, num_classes):
        super(CustomBananaCNN_Variety, self).__init__()
        
        def conv_block(in_channels, out_channels, pool=True, dropout_p=0.0):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            if dropout_p > 0:
                layers.append(nn.Dropout2d(dropout_p))
            return nn.Sequential(*layers)
        
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128, dropout_p=0.1),
            conv_block(128, 256, dropout_p=0.2),
            conv_block(256, 512, dropout_p=0.3),
            conv_block(512, 512, pool=False, dropout_p=0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

class CustomBananaCNN_Ripeness(nn.Module):
    """Custom CNN for banana ripeness detection - NO SOFTMAX for XAI"""
    def __init__(self, num_classes):
        super(CustomBananaCNN_Ripeness, self).__init__()
        
        def conv_block(in_channels, out_channels, pool=True, dropout_p=0.0):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            if dropout_p > 0:
                layers.append(nn.Dropout2d(dropout_p))
            return nn.Sequential(*layers)
        
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128, dropout_p=0.1),
            conv_block(128, 256, dropout_p=0.2),
            conv_block(256, 512, dropout_p=0.3),
            conv_block(512, 512, pool=False, dropout_p=0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

class EfficientNet_Variety(nn.Module):
    """EfficientNet-B0 for variety classification"""
    def __init__(self, num_classes):
        super(EfficientNet_Variety, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNet_Ripeness(nn.Module):
    """EfficientNet-B0 for ripeness detection"""
    def __init__(self, num_classes):
        super(EfficientNet_Ripeness, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class DenseNet_Variety(nn.Module):
    """DenseNet121 for variety classification"""
    def __init__(self, num_classes):
        super(DenseNet_Variety, self).__init__()
        self.backbone = models.densenet121(pretrained=False)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class DenseNet_Ripeness(nn.Module):
    """DenseNet121 for ripeness detection"""
    def __init__(self, num_classes):
        super(DenseNet_Ripeness, self).__init__()
        self.backbone = models.densenet121(pretrained=False)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class VGG16_Variety(nn.Module):
    """VGG16 for variety classification"""
    def __init__(self, num_classes):
        super(VGG16_Variety, self).__init__()
        self.backbone = models.vgg16(pretrained=False)
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class VGG16_Ripeness(nn.Module):
    """VGG16 for ripeness detection"""
    def __init__(self, num_classes):
        super(VGG16_Ripeness, self).__init__()
        self.backbone = models.vgg16(pretrained=False)
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class ViT_Variety(nn.Module):
    """Vision Transformer for variety classification"""
    def __init__(self, num_classes):
        super(ViT_Variety, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class ViT_Ripeness(nn.Module):
    """Vision Transformer for ripeness detection"""
    def __init__(self, num_classes):
        super(ViT_Ripeness, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class DeiT_Variety(nn.Module):
    """DeiT for variety classification"""
    def __init__(self, num_classes):
        super(DeiT_Variety, self).__init__()
        self.backbone = timm.create_model('deit_small_patch16_224', pretrained=False)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class DeiT_Ripeness(nn.Module):
    """DeiT for ripeness detection"""
    def __init__(self, num_classes):
        super(DeiT_Ripeness, self).__init__()
        self.backbone = timm.create_model('deit_small_patch16_224', pretrained=False)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)
