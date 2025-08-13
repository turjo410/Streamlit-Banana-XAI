import os

# === PATHS ===
WEIGHTS_DIR = "./weights"
ASSETS_DIR = "./assets"

# === DEVICE SETUP ===
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === COMPREHENSIVE MODEL CONFIGURATIONS ===
MODELS_CONFIG = {
    "🍌 Variety - Custom CNN": {
        "model_type": "variety",
        "model_class": "CustomBananaCNN_Variety",
        "weight_file": "CustomCNN_variety_best.pth", 
        "classes": ["Bangla Kola", "Champa Kola", "Sabri Kola", "Sagor Kola"],
        "description": "Custom CNN for banana variety classification",
        "input_size": (224, 224),
        "architecture": "Custom CNN - 6 conv layers + 2 FC layers",
        "sample_dir": "variety_samples",
        "num_classes": 4,
        "params": "2.1M",
        "accuracy": "94.2%"
    },
    "🟡 Ripeness - Custom CNN": {
        "model_type": "ripeness",
        "model_class": "CustomBananaCNN_Ripeness",
        "weight_file": "custom_cnn_ripeness_best.pth",
        "classes": ["Green", "Overripe", "Ripe", "Semi-ripe"],
        "description": "Custom CNN for banana ripeness detection",
        "input_size": (224, 224),
        "architecture": "Custom CNN - 6 conv layers + 2 FC layers", 
        "sample_dir": "ripeness_samples",
        "num_classes": 4,
        "params": "2.1M",
        "accuracy": "92.8%"
    },
    "🍌 Variety - EfficientNet-B0": {
        "model_type": "variety",
        "model_class": "EfficientNet_Variety",
        "weight_file": "EfficientNet-B0_variety_classification_best.pth",
        "classes": ["Bangla Kola", "Champa Kola", "Sabri Kola", "Sagor Kola"],
        "description": "EfficientNet-B0 transfer learning for variety classification",
        "input_size": (224, 224),
        "architecture": "EfficientNet-B0 + Custom Classifier",
        "sample_dir": "variety_samples",
        "num_classes": 4,
        "params": "5.3M",
        "accuracy": "97.1%"
    },
    "🟡 Ripeness - EfficientNet-B0": {
        "model_type": "ripeness",
        "model_class": "EfficientNet_Ripeness",
        "weight_file": "EfficientNet-B0_ripeness_detection_best.pth",
        "classes": ["Green", "Overripe", "Ripe", "Semi-ripe"],
        "description": "EfficientNet-B0 for ripeness detection",
        "input_size": (224, 224),
        "architecture": "EfficientNet-B0 + Custom Classifier",
        "sample_dir": "ripeness_samples", 
        "num_classes": 4,
        "params": "5.3M",
        "accuracy": "95.8%"
    },
    "🍌 Variety - DenseNet121": {
        "model_type": "variety",
        "model_class": "DenseNet_Variety",
        "weight_file": "DenseNet121_variety_classification_best.pth",
        "classes": ["Bangla Kola", "Champa Kola", "Sabri Kola", "Sagor Kola"],
        "description": "DenseNet121 transfer learning for variety classification",
        "input_size": (224, 224),
        "architecture": "DenseNet121 + Custom Classifier",
        "sample_dir": "variety_samples",
        "num_classes": 4,
        "params": "7.9M",
        "accuracy": "96.4%"
    },
    "🟡 Ripeness - DenseNet121": {
        "model_type": "ripeness", 
        "model_class": "DenseNet_Ripeness",
        "weight_file": "DenseNet121_ripeness_detection_best.pth",
        "classes": ["Green", "Overripe", "Ripe", "Semi-ripe"],
        "description": "DenseNet121 for ripeness detection",
        "input_size": (224, 224),
        "architecture": "DenseNet121 + Custom Classifier",
        "sample_dir": "ripeness_samples",
        "num_classes": 4,
        "params": "7.9M",
        "accuracy": "94.6%"
    },
    "🍌 Variety - VGG16": {
        "model_type": "variety",
        "model_class": "VGG16_Variety",
        "weight_file": "VGG16_variety_classification_best.pth",
        "classes": ["Bangla Kola", "Champa Kola", "Sabri Kola", "Sagor Kola"],
        "description": "VGG16 transfer learning for variety classification",
        "input_size": (224, 224),
        "architecture": "VGG16 + Custom Classifier",
        "sample_dir": "variety_samples",
        "num_classes": 4,
        "params": "138.4M",
        "accuracy": "95.7%"
    },
    "🟡 Ripeness - VGG16": {
        "model_type": "ripeness",
        "model_class": "VGG16_Ripeness", 
        "weight_file": "VGG16_ripeness_detection_best.pth",
        "classes": ["Green", "Overripe", "Ripe", "Semi-ripe"],
        "description": "VGG16 for ripeness detection",
        "input_size": (224, 224),
        "architecture": "VGG16 + Custom Classifier",
        "sample_dir": "ripeness_samples",
        "num_classes": 4,
        "params": "138.4M",
        "accuracy": "93.2%"
    },
    "🍌 Variety - ViT-Base-16": {
        "model_type": "variety",
        "model_class": "ViT_Variety",
        "weight_file": "ViT-Base-16_variety_classification_best.pth",
        "classes": ["Bangla Kola", "Champa Kola", "Sabri Kola", "Sagor Kola"],
        "description": "Vision Transformer for variety classification",
        "input_size": (224, 224),
        "architecture": "ViT-Base-16 + Custom Head",
        "sample_dir": "variety_samples",
        "num_classes": 4,
        "params": "86.6M",
        "accuracy": "98.1%"
    },
    "🟡 Ripeness - ViT-Base-16": {
        "model_type": "ripeness",
        "model_class": "ViT_Ripeness",
        "weight_file": "ViT-Base-16_ripeness_detection_best.pth", 
        "classes": ["Green", "Overripe", "Ripe", "Semi-ripe"],
        "description": "Vision Transformer for ripeness detection",
        "input_size": (224, 224),
        "architecture": "ViT-Base-16 + Custom Head",
        "sample_dir": "ripeness_samples",
        "num_classes": 4,
        "params": "86.6M",
        "accuracy": "96.9%"
    },
    "🍌 Variety - DeiT-Small-16": {
        "model_type": "variety",
        "model_class": "DeiT_Variety",
        "weight_file": "DeiT-Small-16_variety_classification_best.pth",
        "classes": ["Bangla Kola", "Champa Kola", "Sabri Kola", "Sagor Kola"],
        "description": "DeiT Small for variety classification",
        "input_size": (224, 224),
        "architecture": "DeiT-Small-16 + Custom Head",
        "sample_dir": "variety_samples",
        "num_classes": 4,
        "params": "22.1M",
        "accuracy": "97.8%"
    },
    "🟡 Ripeness - DeiT-Small-16": {
        "model_type": "ripeness",
        "model_class": "DeiT_Ripeness",
        "weight_file": "DeiT-Small-16_ripeness_detection_best.pth",
        "classes": ["Green", "Overripe", "Ripe", "Semi-ripe"],
        "description": "DeiT Small for ripeness detection",
        "input_size": (224, 224),
        "architecture": "DeiT-Small-16 + Custom Head",
        "sample_dir": "ripeness_samples",
        "num_classes": 4,
        "params": "22.1M",
        "accuracy": "95.4%"
    }
}

# === IMAGE SETTINGS ===
IMG_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# === XAI METHODS ===
XAI_METHODS = ["Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Ablation-CAM", "LIME"]
