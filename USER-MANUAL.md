# 🍌 User Manual: Banana XAI Classification App

## 📋 Table of Contents
1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Step-by-Step Usage](#step-by-step-usage)
4. [Understanding Results](#understanding-results)
5. [XAI Explanations Guide](#xai-explanations-guide)
6. [Troubleshooting](#troubleshooting)
7. [Tips for Best Results](#tips-for-best-results)

## 🚀 Getting Started

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: At least 4GB
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)
- **Internet**: Connection for initial setup

### Installation Steps
1. **Download** the app from the provided GitHub repository
2. **Install dependencies**: Run `pip install -r requirements.txt`
3. **Place model weights** in the `weights/` folder
4. **Start the app**: Run `streamlit run app.py`
5. **Access the app**: Open your browser to `http://localhost:8501`

## 🎨 Interface Overview

### Main Components

#### 🎯 Header Section
- Application title and mode indicator (Variety/Ripeness Analysis)
- Professional branding and navigation

#### 📋 Sidebar Controls
- **Analysis Mode**: Toggle between Variety and Ripeness classification
- **Model Selection**: Dropdown with 12 available AI models
- **Model Info**: Architecture, accuracy, and parameter details
- **Image Input**: Upload or sample selection options

#### 📊 Main Display Area
- **Input Image**: Preview of selected/uploaded image with properties
- **Analysis Results**: Prediction results with confidence scores
- **XAI Visualizations**: 5 explanation methods side-by-side
- **Download Options**: Export results as ZIP file

## 📝 Step-by-Step Usage

### Step 1: Choose Analysis Mode

1. **Open the app** in your browser
2. **Select mode** from the sidebar:
   - 🍌 **Variety Analysis**: Classify banana types (Sagor, Champa, Bangla, Sabri)
   - 🟡 **Ripeness Analysis**: Determine ripeness level (Green, Semi-ripe, Ripe, Overripe)

### Step 2: Select an AI Model

1. **Click the model dropdown** in the sidebar
2. **Choose from 12 models**:
   - Custom CNN (lightweight, fast)
   - EfficientNet (balanced performance)
   - DenseNet (high accuracy)
   - VGG16 (classic architecture)
   - ViT (Vision Transformer)
   - DeiT (efficient transformer)
3. **View model details**: Architecture info, accuracy, and parameters appear below

### Step 3: Input Your Image

#### Option A: Upload Image
1. **Click "Browse files"** in the sidebar
2. **Select a JPG/PNG image** of a banana
3. **Image preview** appears in the main area

#### Option B: Use Sample Images
1. **Select "📁 Choose from samples"** radio button
2. **Pick from provided examples** using the dropdown
3. **Sample image loads** automatically

### Step 4: Run Analysis

1. **Image properties display**: Dimensions, size, and format
2. **Analysis starts automatically** when image is selected
3. **Progress indicators** show processing status
4. **Results appear** in organized sections

### Step 5: Review Results

1. **Main prediction** with confidence percentage
2. **Top-3 predictions** with detailed breakdown
3. **Model performance metrics**
4. **XAI visualizations** in grid layout

### Step 6: Download Results (Optional)

1. **Click download link** in the results section
2. **ZIP file contains**:
   - Original image
   - All XAI visualization overlays
   - Results summary

## 📊 Understanding Results

### Prediction Output

#### 🏆 Main Prediction
- **Class Name**: The model's top prediction
- **Confidence**: Percentage certainty (higher = more confident)
- **Working Methods**: Number of XAI methods successfully generated (should be 5/5)

#### 📈 Detailed Results
- **Rank**: #1, #2, #3 predictions
- **Class Names**: Specific variety or ripeness level
- **Probability**: Likelihood for each class
- **Progress Bars**: Visual confidence indicators

#### 📋 Model Performance
- **Accuracy**: Model's training/validation accuracy
- **Parameters**: Model complexity (fewer = faster)
- **Classes**: Number of categories the model can predict

### Confidence Interpretation

| Range | Interpretation |
|-------|---------------|
| 90-100% | Very confident prediction |
| 70-89% | Confident prediction |
| 50-69% | Moderate confidence |
| Below 50% | Low confidence, consider different image |

## 🔍 XAI Explanations Guide

### Understanding Visual Explanations

#### 🎯 Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Shows**: Which image regions influenced the prediction most
- **Colors**: Red/Yellow = high importance, Blue = low importance
- **Best for**: Understanding model focus areas

#### 🎯 Grad-CAM++
- **Shows**: Improved version of Grad-CAM with better localization
- **Colors**: Similar to Grad-CAM but more precise
- **Best for**: Detailed region analysis

#### 🎯 Eigen-CAM
- **Shows**: Principal component analysis of activations
- **Colors**: Highlights most significant feature patterns
- **Best for**: Understanding feature importance

#### 🎯 Ablation-CAM
- **Shows**: Impact of removing different image regions
- **Colors**: Areas crucial for maintaining prediction
- **Best for**: Critical region identification

#### 🎯 LIME (Local Interpretable Model-agnostic Explanations)
- **Shows**: Local decision boundaries around the prediction
- **Colors**: Green = supports prediction, Red = opposes prediction
- **Best for**: Understanding local decision making

### Reading the Visualizations
- **Brighter colors** = higher importance
- **Overlaid on original** = easy comparison
- **Side-by-side layout** = compare different methods
- **Consistent predictions** across methods = reliable results

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### ❌ App won't start
**Solution:**
1. Check Python version: `python --version` (need 3.8+)
2. Reinstall requirements: `pip install -r requirements.txt`
3. Try: `streamlit run app.py --server.port 8501`

#### ❌ "Model not found" error
**Solution:**
1. Check `weights/` folder exists
2. Verify `.pth` files are present
3. Ensure file names match `config.py` exactly
4. Download missing weights from provided source

#### ❌ Upload not working
**Solution:**
1. Use JPG or PNG files only
2. File size should be < 200MB
3. Try different browser
4. Clear browser cache

#### ❌ Poor predictions
**Solution:**
1. Use clear, well-lit images
2. Ensure banana fills most of frame
3. Avoid blurry or low-resolution images
4. Try different models for comparison

#### ❌ XAI methods failing
**Solution:**
1. This is normal occasionally (model architecture dependent)
2. Try different target layers
3. Some methods work better with specific models
4. 3-4 working methods is still good

#### ❌ Slow performance
**Solution:**
1. Use smaller images (app auto-resizes)
2. Close other browser tabs
3. Restart the application
4. Check system memory usage

### Getting Help
- Check error messages in the debug expander
- Try different images to isolate issues
- Restart the app if behavior seems odd
- Contact team lead for persistent problems

## 💡 Tips for Best Results

### Image Selection

#### ✅ Good Images:
- Clear, well-lit bananas
- Single banana in frame
- Minimal background clutter
- High resolution (app handles resizing)
- Natural colors

#### ❌ Avoid:
- Multiple overlapping bananas
- Very dark or overexposed images
- Bananas too small in frame
- Heavy filters or artificial colors
- Blurry or low-quality photos

### Model Selection
- **For Speed**: Custom CNN (fastest, decent accuracy)
- **For Accuracy**: ViT or EfficientNet (slower but most accurate)
- **For Balance**: DenseNet or EfficientNet-B0
- **For Comparison**: Try multiple models on same image

### Interpreting Results
- **High confidence + consistent XAI** = reliable prediction
- **Low confidence** = try different image or model
- **Inconsistent XAI patterns** = model uncertainty
- **Multiple methods agreeing** = strong evidence

### Using XAI Effectively
- **Compare all 5 methods** for complete picture
- **Look for consistent patterns** across visualizations
- **Bright red/yellow areas** in heatmaps = most important
- **LIME green regions** support the prediction

## 👥 Support Team

### Group Members
- **Shahriar Khan** (2022-3-60-016): Lead Developer, Contribution: 40%
- **Tanvir Rahman** (2022-3-60-134): Model Trainer, Contribution: 30%
- **Khalid Mahmud Joy** (2022-3-60-016): App Developer, Contribution: 20%
- **Rifah Tamanna** (2022-3-60-016): Documentation & Testing, Contribution: 10%

### Contact Information
For technical support or questions:
- **Primary Contact**: Shahriar Khan
- **Email**: [turjo410@gmail.com](mailto:turjo410@gmail.com)
- **Secondary**: Any team member for specific issues

## 📝 Quick Reference

### Keyboard Shortcuts
- **Ctrl + R**: Refresh the app
- **F5**: Reload page
- **Ctrl + Shift + R**: Hard refresh (clears cache)

### File Formats
- **Supported**: JPG, JPEG, PNG
- **Max Size**: 200MB
- **Recommended**: 1-5MB for best performance

### Expected Performance
- **Upload Time**: < 5 seconds
- **Analysis Time**: 10-30 seconds depending on model
- **XAI Generation**: 5-15 seconds per method
- **Total Time**: 30-60 seconds for complete analysis

---

<div align="center">




</div>
