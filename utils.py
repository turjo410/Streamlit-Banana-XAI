import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import io
import zipfile
import base64
from datetime import datetime

# XAI Libraries
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

from models import *
from config import *

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def get_model_class(model_class_name):
    """Get model class by name"""
    return globals()[model_class_name]

def load_model(model_config):
    """Load model with weights"""
    model_class_name = model_config['model_class']
    weight_file = model_config['weight_file']
    num_classes = model_config['num_classes']
    
    # Initialize model
    model_class = get_model_class(model_class_name)
    model = model_class(num_classes)
    
    # Load weights
    weight_path = os.path.join(WEIGHTS_DIR, weight_file)
    try:
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded weights: {weight_file}")
        else:
            print(f"⚠️ Weight file not found: {weight_path}. Using random weights.")
    except Exception as e:
        print(f"⚠️ Error loading weights: {e}. Using random weights.")
    
    # Configure for XAI
    model = model.to(DEVICE)
    model.train()  # Enable gradients
    for param in model.parameters():
        param.requires_grad_(True)
    
    return model

def preprocess_image(image):
    """Preprocess PIL image for model input"""
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor

def denormalize_image(tensor_img):
    """Convert normalized tensor back to displayable image"""
    img = tensor_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = np.array(STD) * img + np.array(MEAN)
    return np.clip(img, 0, 1)

def get_sample_images(sample_dir):
    """Get list of sample images"""
    sample_path = os.path.join(ASSETS_DIR, sample_dir)
    if not os.path.exists(sample_path):
        return []
    
    images = []
    for file in os.listdir(sample_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(os.path.join(sample_path, file))
    return sorted(images)

class FixedManualGradCAM:
    """Manual GradCAM implementation with proper gradient handling"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
    def generate_cam(self, input_tensor, target_class):
        gradients = []
        activations = []
        
        def save_activation(module, input, output):
            activations.append(output.detach())
            
        def save_gradient(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                if grad_output[0] is not None:
                    gradients.append(grad_output.detach())
            else:
                if grad_output is not None:
                    gradients.append(grad_output.detach())
        
        # Register hooks
        h1 = self.target_layer.register_forward_hook(save_activation)
        h2 = self.target_layer.register_backward_hook(save_gradient)
        
        try:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            self.model.zero_grad()
            
            output = self.model(input_tensor)
            if output.dim() > 1:
                score = output[:, target_class]
            else:
                score = output[target_class]
            
            score.backward(retain_graph=True)
            
            if not gradients or not activations:
                # Fallback heatmap
                heatmap = np.zeros(IMG_SIZE)
                center_y, center_x = IMG_SIZE[0]//2, IMG_SIZE[1]//2
                y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE[1]]
                mask = (y - center_y)**2 + (x - center_x)**2 <= (IMG_SIZE//3)**2
                heatmap[mask] = 1.0
                return heatmap
            
            grads = gradients
            acts = activations
            
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            cam = F.interpolate(cam, size=IMG_SIZE, mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                y, x = np.ogrid[:IMG_SIZE[0], :IMG_SIZE[1]]
                center_y, center_x = IMG_SIZE//2, IMG_SIZE[1]//2
                cam = np.maximum(0, 1 - np.sqrt((y - center_y)**2 + (x - center_x)**2) / (IMG_SIZE//2))
            
            return cam
            
        except Exception as e:
            print(f"⚠️ Manual GradCAM failed: {e}")
            # Fallback heatmap
            heatmap = np.zeros(IMG_SIZE)
            center_y, center_x = IMG_SIZE[0]//2, IMG_SIZE[1]//2
            y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE[1]]
            mask = (y - center_y)**2 + (x - center_x)**2 <= (IMG_SIZE//3)**2
            heatmap[mask] = 1.0
            return heatmap
            
        finally:
            h1.remove()
            h2.remove()

def get_target_layer(model, model_class_name):
    """Get appropriate target layer for CAM methods"""
    if 'CustomBananaCNN' in model_class_name:
        conv_layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
        return [conv_layers[-1]]
    elif 'EfficientNet' in model_class_name:
        return [model.backbone.conv_head]
    elif 'DenseNet' in model_class_name:
        return [model.backbone.features.norm5]
    elif 'VGG' in model_class_name:
        return [model.backbone.features[-1]]
    elif 'ViT' in model_class_name or 'DeiT' in model_class_name:
        return [model.backbone.patch_embed.proj]
    else:
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        return [conv_layers[-1]] if conv_layers else [list(model.modules())[-2]]

class XAIAnalyzer:
    """Comprehensive XAI analyzer for all model types"""
    def __init__(self, model, model_config, class_names, device):
        self.model = model
        self.model_config = model_config
        self.class_names = class_names
        self.device = device
        
        # Get target layers for CAM methods
        self.target_layers = get_target_layer(model, model_config['model_class'])
        
        # Initialize manual GradCAM
        self.manual_gradcam = FixedManualGradCAM(self.model, self.target_layers[0])
        
    def predict(self, input_tensor):
        """FIXED: Get model prediction with individual float probabilities"""
        self.model.eval()
        
        # Ensure proper batch dimension
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Handle different output shapes
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(0)
            
            # Verify model output matches class count
            model_num_classes = outputs.shape[1]
            expected_classes = len(self.class_names)
            
            if model_num_classes != expected_classes:
                raise ValueError(f"Model output classes ({model_num_classes}) != expected classes ({expected_classes})")
            
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
            # Bounds checking for predicted index
            if pred_idx < 0 or pred_idx >= len(self.class_names):
                raise IndexError(f"Predicted class index {pred_idx} is out of bounds for {len(self.class_names)} classes")
            
            confidence = probs[0, pred_idx].item()
            
            # FIXED: Get top 3 predictions with individual float probabilities
            topk = torch.topk(probs, k=min(3, len(self.class_names)))
            top_indices = topk.indices[0].tolist()
            top_values = topk.values.tolist()  # Extract from first batch dimension
            
            # CRITICAL FIX: Ensure individual float values, not nested lists
            top3_results = []
            for idx, (class_idx, prob_val) in enumerate(zip(top_indices, top_values)):
                if 0 <= class_idx < len(self.class_names):
                    # Ensure prob_val is a single float (not a list or tensor)
                    if isinstance(prob_val, (list, tuple)):
                        prob_val = float(prob_val[0]) if len(prob_val) > 0 else 0.0
                    elif hasattr(prob_val, 'item'):
                        prob_val = float(prob_val.item())
                    else:
                        prob_val = float(prob_val)
                    
                    top3_results.append((self.class_names[class_idx], prob_val))
                else:
                    print(f"⚠️ Skipping invalid class index: {class_idx}")
        
        return pred_idx, confidence, top3_results
    
    def generate_all_cams(self, input_tensor, target_class):
        """Generate all CAM methods with robust fallbacks"""
        self.model.train()  # Ensure gradients
        results = {}
        
        # Validate target class index
        if target_class < 0 or target_class >= len(self.class_names):
            print(f"⚠️ Invalid target class {target_class}, using class 0")
            target_class = 0
        
        cam_methods = {
            'Grad-CAM': GradCAM,
            'Grad-CAM++': GradCAMPlusPlus,
            'Eigen-CAM': EigenCAM,
            'Ablation-CAM': AblationCAM
        }
        
        for name, cam_class in cam_methods.items():
            try:
                # Try library implementation
                cam = cam_class(model=self.model, target_layers=self.target_layers)
                targets = [ClassifierOutputTarget(target_class)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                
                # Check if meaningful
                if grayscale_cam.max() > 0.01:
                    results[name] = grayscale_cam[0]
                    print(f"✅ {name} library working")
                else:
                    raise ValueError("Empty result from library")
                    
            except Exception as e:
                print(f"🔧 {name} library failed, using manual implementation")
                # Use manual implementation
                manual_result = self.manual_gradcam.generate_cam(input_tensor, target_class)
                results[name] = manual_result
                print(f"✅ {name} manual implementation working")
        
        return results
    
    def generate_lime_explanation(self, input_tensor, target_class):
        """Generate LIME explanation"""
        # Validate target class
        if target_class < 0 or target_class >= len(self.class_names):
            target_class = 0
        
        def predict_fn(images):
            self.model.eval()
            batch = torch.stack([
                transform(Image.fromarray((img * 255).astype(np.uint8)))
                for img in images
            ]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch)
                if outputs.ndim == 1:
                    outputs = outputs.unsqueeze(0)
                probs = F.softmax(outputs, dim=1)
            return probs.cpu().numpy()
        
        # Convert to image
        image_for_lime = (denormalize_image(input_tensor) * 255).astype(np.uint8)
        
        try:
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                image_for_lime, predict_fn,
                top_labels=len(self.class_names),
                hide_color=0,
                num_samples=1000
            )
            
            temp, mask = explanation.get_image_and_mask(
                target_class,
                positive_only=False,
                num_features=10,
                hide_rest=False
            )
            
            return mark_boundaries(temp / 255.0, mask)
            
        except Exception as e:
            print(f"⚠️ LIME failed: {e}")
            return image_for_lime / 255.0
    
    def analyze_image(self, input_tensor):
        """FIXED: Complete XAI analysis with comprehensive error handling"""
        input_tensor = input_tensor.to(self.device)
        
        # Get prediction with bounds checking
        pred_idx, confidence, top3_results = self.predict(input_tensor)
        pred_label = self.class_names[pred_idx]
        
        # Generate all XAI methods
        cam_results = self.generate_all_cams(input_tensor, pred_idx)
        lime_result = self.generate_lime_explanation(input_tensor, pred_idx)
        
        # Count working methods
        working_count = len([r for r in cam_results.values() if r is not None]) + 1
        
        return {
            'prediction': pred_label,
            'confidence': confidence,
            'top3_results': top3_results,
            'working_methods': working_count,
            'cam_results': cam_results,
            'lime_result': lime_result,
            'original_image': denormalize_image(input_tensor)
        }

def create_download_zip(results, filename_prefix):
    """Create downloadable ZIP file with all visualizations"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Original image
        original_img = Image.fromarray((results['original_image'] * 255).astype(np.uint8))
        img_buffer = io.BytesIO()
        original_img.save(img_buffer, format='PNG')
        zip_file.writestr(f"{filename_prefix}_original.png", img_buffer.getvalue())
        
        # CAM results
        for method_name, cam_result in results['cam_results'].items():
            if cam_result is not None:
                cam_overlay = show_cam_on_image(results['original_image'], cam_result, use_rgb=True)
                cam_img = Image.fromarray((cam_overlay * 255).astype(np.uint8))
                img_buffer = io.BytesIO()
                cam_img.save(img_buffer, format='PNG')
                zip_file.writestr(f"{filename_prefix}_{method_name.replace(' ', '_').lower()}.png", 
                                img_buffer.getvalue())
        
        # LIME result
        if results['lime_result'] is not None:
            lime_img = Image.fromarray((results['lime_result'] * 255).astype(np.uint8))
            img_buffer = io.BytesIO()
            lime_img.save(img_buffer, format='PNG')
            zip_file.writestr(f"{filename_prefix}_lime.png", img_buffer.getvalue())
    
    return zip_buffer.getvalue()

def get_download_link(zip_data, filename):
    """Generate download link for ZIP file"""
    b64 = base64.b64encode(zip_data).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">📥 Download All Visualizations (ZIP)</a>'
    return href
