import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go
import base64
import io

# Import custom modules
from utils import *
from config import *

# === STREAMLIT CONFIGURATION ===
st.set_page_config(
    page_title="🍌 Banana Custom CNN XAI Assignment",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === GUARANTEED FIXED SIDEBAR & PROFESSIONAL THEME ===
# FIXED: Updated CSS for button layout and appearance
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700;800;900&display=swap');
    
    /* Hide Streamlit branding and warnings */
    #MainMenu, header, footer, .stDeployButton {
        visibility: hidden;
    }
    
    /* FIXED: Apply Inter font only to text elements, NOT icons */
    body, html, p, h1, h2, h3, h4, h5, h6, label, strong, em, a,
    .stMarkdown, .stText, .stSelectbox label, .stRadio label, .stFileUploader label {
        color: white !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Keep white color for everything else but don't force font */
    *, *::before, *::after, div, span {
        color: white !important;
    }
    
    /* Global dark background */
    .stApp, .main, body, html {
        background-color: #0f172a !important;
        color: white !important;
    }
    
    /* --- START: GUARANTEED FIXED SIDEBAR CSS --- */
    section[data-testid="stSidebar"] {
        position: fixed !important;
        left: 0px !important;
        top: 0px !important;
        height: 100% !important;
        width: 350px !important;
        min-width: 350px !important;
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        z-index: 999999 !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5) !important;
        border-right: 3px solid #334155 !important;
        transform: none !important;
        transition: none !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding: 2rem !important;
    }
    button[kind="header"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    div.block-container {
        margin-left: 370px !important;
        padding: 2rem 3rem !important;
        width: calc(100% - 370px) !important;
        max-width: 1400px !important;
    }
    /* --- END: GUARANTEED FIXED SIDEBAR CSS --- */
    
    /* Professional main title */
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        margin: 2rem 0;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.25rem;
        color: #94a3b8 !important;
        margin-bottom: 3rem;
        font-weight: 500;
    }
    
    /* --- START: ULTIMATE FIXED BUTTONS --- */
    /* General style for the large professional buttons on the landing page */
    .stButton > button {
        width: 100% !important; /* Make button fill its column */
        height: 250px !important; /* Increased height */
        font-size: 2.8rem !important; /* Increased font size */
        font-weight: 800 !important;
        font-family: 'Poppins', sans-serif !important;
        line-height: 1.25 !important; /* Adjust for multi-line text */
        border-radius: 20px !important;
        border: none !important;
        margin: 1rem 0 !important; /* Vertical margin */
        transition: all 0.3s ease !important;
        box-shadow: 0 12px 30px rgba(0,0,0,0.4) !important;
        color: white !important;
        display: flex !important; /* Enable flexbox for alignment */
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        padding: 1rem !important;
    }
    
    /* Enhanced hover effect for attractiveness */
    .stButton > button:hover {
        transform: translateY(-8px) scale(1.03) !important; /* More pronounced effect */
        box-shadow: 0 20px 45px rgba(0,0,0,0.55) !important;
    }
    
    /* FIXED: Target the second column for the first button's style (Variety) */
    div[data-testid="column"]:nth-child(2) .stButton > button {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    }
    
    /* FIXED: Target the third column for the second button's style (Ripeness) */
    div[data-testid="column"]:nth-child(3) .stButton > button {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
    }
    /* --- END: ULTIMATE FIXED BUTTONS --- */

    /* Professional Task Header */
    .task-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }
    
    .task-header * {
        color: white !important;
    }
    
    /* Sidebar Headers */
    .sidebar-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
    }
    
    .sidebar-header * {
        color: white !important;
    }
    
    /* Prediction Widget */
    .prediction-widget {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 3rem 0;
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.4);
    }
    
    .prediction-widget * {
        color: white !important;
    }
    
    .prediction-widget h2 {
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 1.5rem;
    }
    
    /* Image Properties Widget */
    .image-properties {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.4);
    }
    
    .image-properties * {
        color: white !important;
    }
    
    /* Prediction Items */
    .prediction-item {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .prediction-item * {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 3rem 0;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Professional images */
    .stImage img {
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

def show_landing_page():
    """Professional landing page with fixed sidebar and white fonts"""
    # Enhanced sidebar with white text - always visible
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2 style="margin: 0; font-size: 1.5rem; font-family: 'Poppins', sans-serif;">🍌 Banana AI Explorer</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">Professional Analysis Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("**Step 1:** Select your analysis task")
        st.markdown("**Step 2:** Choose an AI model")
        st.markdown("**Step 3:** Upload or select an image")
        st.markdown("**Step 4:** Get instant results with explanations")
        
        st.markdown("---")
        st.markdown("### 🎯 Available Tasks")
        st.markdown("**🍌 Variety Classification**")
        st.markdown("- Bangla Kola, Champa Kola")
        st.markdown("- Sabri Kola, Sagor Kola")
        
        st.markdown("**🟡 Ripeness Detection**")
        st.markdown("- Green, Semi-ripe")
        st.markdown("- Ripe, Overripe")
        
        st.markdown("---")
        st.markdown("### 🤖 AI Models Available")
        st.markdown("- **Custom CNN** (Fast)")
        st.markdown("- **EfficientNet-B0** (Balanced)")
        st.markdown("- **DenseNet121** (Accurate)")
        st.markdown("- **VGG16** (Robust)")
        st.markdown("- **ViT-Base-16** (State-of-art)")
        st.markdown("- **DeiT-Small-16** (Efficient)")
        
        st.markdown("---")
        st.markdown("### 📊 XAI Methods")
        st.markdown("- Grad-CAM")
        st.markdown("- Grad-CAM++")
        st.markdown("- Eigen-CAM")
        st.markdown("- Ablation-CAM")
        st.markdown("- LIME")
    
    # Main title
    st.markdown('<h1 class="main-title">Banana Custom CNN XAI Assignment</h1>', unsafe_allow_html=True)
    
    # Subtitle
    st.markdown('<div class="subtitle">Advanced AI-powered banana analysis with explainable artificial intelligence</div>', unsafe_allow_html=True)
    
    # --- START: ULTIMATE FIXED LAYOUT ---
    # FIXED: Use a 4-column layout to create margins and center the main buttons.
    # The layout is [spacer, button1, button2, spacer] for perfect alignment.
    _ , col1, col2, _ = st.columns([0.5, 1.5, 1.5, 0.5], gap="large")
    # --- END: ULTIMATE FIXED LAYOUT ---
    
    with col1:
        if st.button("🍌 VARIETY\nCLASSIFICATION\n\nIdentify banana varieties", 
                     key="variety_btn", 
                     help="Classify different banana varieties using AI models"):
            st.session_state.selected_task = "variety"
            st.rerun()
    
    with col2:
        if st.button("🟡 RIPENESS\nDETECTION\n\nAssess banana ripeness", 
                     key="ripeness_btn", 
                     help="Detect banana ripeness stages using AI models"):
            st.session_state.selected_task = "ripeness"
            st.rerun()

def show_main_interface():
    """Professional main interface with fixed sidebar and white fonts"""
    task = st.session_state.selected_task
    
    # Task header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div class="task-header">
            <h2>🎯 {task.title()} Analysis Mode - Professional AI Interface</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Reset", key="reset_btn", help="Return to task selection"):
            del st.session_state.selected_task
            st.rerun()
    
    # Filter models
    available_models = {k: v for k, v in MODELS_CONFIG.items() 
                        if v['model_type'] == task}
    
    # Fixed sidebar with model selection
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-header">
            <h3 style="margin-bottom: 1rem; font-size: 1.25rem;">🔧 Model Selection</h3>
            <p style="font-size: 1rem;">Task: <strong>{task.title()}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        selected_model_name = st.selectbox(
            "Choose AI Model:",
            list(available_models.keys()),
            help="Select from trained models for your task"
        )
        
        model_config = available_models[selected_model_name]
        
        # Model information
        with st.expander("📊 Model Details", expanded=True):
            st.markdown(f"""
            **Architecture:** {model_config['architecture']}  
            **Parameters:** {model_config['params']}  
            **Accuracy:** {model_config['accuracy']}  
            **Classes:** {model_config['num_classes']}  
            **Input Size:** {model_config['input_size']}
            """)
        
        # Image input
        st.markdown("### 📸 Image Input")
        input_method = st.radio(
            "Input Method:",
            ["Upload Image", "Select Sample"],
            help="Choose your input method"
        )
        
        image = None
        image_source = ""
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload banana image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image for analysis"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                image_source = uploaded_file.name
                st.success(f"✅ Uploaded: {uploaded_file.name}")
            else:
                st.info("👆 Upload an image to start analysis")
                return
        
        else:
            sample_images = get_sample_images(model_config['sample_dir'])
            
            if sample_images:
                selected_sample = st.selectbox(
                    "Choose sample:",
                    [os.path.basename(img) for img in sample_images]
                )
                
                sample_path = next(img for img in sample_images 
                                   if os.path.basename(img) == selected_sample)
                image = Image.open(sample_path).convert('RGB')
                image_source = selected_sample
                st.success(f"✅ Selected: {selected_sample}")
            else:
                st.error("No sample images found")
                return
    
    # Main content
    if image is not None:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        
        # Layout
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            st.markdown("### 📷 Input Image")
            st.image(image, caption=f"Source: {image_source}", use_container_width=True)
            
            # Image Properties Widget - FIXED the array shape access
            img_array = np.array(image)
            st.markdown(f"""
            <div class="image-properties">
                <h3>📏 Image Properties</h3>
                <p><strong>Dimensions:</strong> {img_array.shape[1]} × {img_array.shape[0]}</p>
                <p><strong>Channels:</strong> {img_array.shape[2]}</p>
                <p><strong>Size:</strong> {img_array.nbytes / 1024:.1f} KB</p>
                <p><strong>Format:</strong> RGB Color</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎯 AI Analysis Results")
            
            # Run analysis
            with st.spinner("🔄 Running AI analysis and generating explanations..."):
                try:
                    # Load model
                    model = load_model(model_config)
                    
                    # Create analyzer
                    analyzer = XAIAnalyzer(model, model_config, model_config['classes'], DEVICE)
                    
                    # Preprocess and analyze
                    input_tensor = preprocess_image(image)
                    results = analyzer.analyze_image(input_tensor)
                    
                    # Prediction Display Widget
                    st.markdown(f"""
                    <div class="prediction-widget">
                        <h2>🏆 Prediction: {results['prediction']}</h2>
                        <h3>📊 Confidence: {results['confidence']:.2%}</h3>
                        <p>✅ XAI Methods Working: {results['working_methods']}/5</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top-3 predictions - FIXED the index access
                    st.markdown('<h3 class="top3-header">📊 Detailed Results</h3>', unsafe_allow_html=True)
                    
                    if results.get('top3_results'):
                        for i, prediction_item in enumerate(results['top3_results']):
                            if isinstance(prediction_item, (list, tuple)) and len(prediction_item) >= 2:
                                class_name = str(prediction_item[0])
                                prob_data = prediction_item[1]
                                
                                # Safe float conversion
                                try:
                                    if hasattr(prob_data, 'item'):
                                        prob_value = float(prob_data.item())
                                    elif isinstance(prob_data, (list, tuple)):
                                        prob_value = float(prob_data) if len(prob_data) > 0 else 0.0
                                    elif isinstance(prob_data, (int, float)):
                                        prob_value = float(prob_data)
                                    else:
                                        prob_value = float(prob_data)
                                    prob_value = max(0.0, min(1.0, prob_value))
                                except:
                                    prob_value = 0.0
                                
                                # Display with white text
                                st.markdown(f"""
                                <div class="prediction-item">
                                    <strong>#{i+1}: {class_name}</strong><br>
                                    <span>Confidence: {prob_value:.2%}</span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.progress(prob_value, text=f"{class_name}: {prob_value:.1%}")
                                
                                if i < len(results['top3_results']) - 1:
                                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Model performance
                    st.markdown("### 📈 Model Performance")
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    
                    with perf_col1:
                        st.metric("Accuracy", model_config['accuracy'], "High Performance")
                    with perf_col2:
                        st.metric("Parameters", model_config['params'], "Optimized")
                    with perf_col3:
                        st.metric("Classes", model_config['num_classes'], f"{task.title()}")
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.info("💡 Try uploading a different image or selecting a sample")
                    return
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # XAI Visualizations
        st.markdown("---")
        st.markdown("## 🔍 Explainable AI Visualizations")
        st.markdown("*Understanding AI decision-making through multiple explanation methods*")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Grad-CAM", "⭐ Grad-CAM++", "🧠 Eigen-CAM", 
            "🔬 Ablation-CAM", "🎨 LIME"
        ])
        
        tabs_info = {
            tab1: ("Grad-CAM", "Gradient-weighted Class Activation Mapping highlights important regions using gradients"),
            tab2: ("Grad-CAM++", "Enhanced Grad-CAM with improved localization capabilities for better accuracy"),
            tab3: ("Eigen-CAM", "Principal component analysis of activation maps for feature importance"),
            tab4: ("Ablation-CAM", "Systematic feature removal analysis to determine contribution importance"),
            tab5: ("LIME", "Local interpretable explanations with green/red regions showing support/opposition")
        }
        
        for tab, (method, description) in tabs_info.items():
            with tab:
                col1, col2 = st.columns(2, gap="medium")
                
                with col1:
                    st.markdown("**Original Image**")
                    if 'results' in locals():
                        st.image(results['original_image'], use_container_width=True)
                
                with col2:
                    st.markdown(f"**{method} Explanation**")
                    
                    if 'results' in locals():
                        if method == "LIME":
                            if results.get('lime_result') is not None:
                                st.image(results['lime_result'], use_container_width=True)
                                st.success("✅ LIME visualization generated successfully")
                            else:
                                st.error("❌ LIME visualization failed")
                        else:
                            cam_key = method.replace('-', '_').replace(' ', '_').lower()
                            cam_result = None
                            
                            if 'cam_results' in results:
                                cam_result = (results['cam_results'].get(cam_key) or 
                                              results['cam_results'].get(method))
                            
                            if cam_result is not None:
                                try:
                                    cam_overlay = show_cam_on_image(
                                        results['original_image'], 
                                        cam_result, 
                                        use_rgb=True
                                    )
                                    st.image(cam_overlay, use_container_width=True)
                                    st.success(f"✅ {method} visualization generated successfully")
                                except Exception as e:
                                    st.error(f"❌ Error displaying {method}: {e}")
                            else:
                                st.error(f"❌ {method} visualization failed")
                
                # Method description
                st.markdown(f"""
                <div class="method-description">
                    <strong>{method} Overview:</strong><br>
                    {description}
                </div>
                """, unsafe_allow_html=True)
        
        # Download Section
        st.markdown("---")
        st.markdown("""
        <div class="download-section">
            <h3 style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem;">💾 Export Analysis Results</h3>
            <p style="font-size: 1rem; margin-bottom: 1.5rem;">Download complete analysis package with all visualizations</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📥 **Download Complete Analysis Package**", 
                      type="primary", 
                      use_container_width=True,
                      help="Download all XAI visualizations and results as ZIP file"):
            if 'results' in locals():
                with st.spinner("Creating download package..."):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_prefix = f"{task}_{selected_model_name.replace(' ', '_')}_{timestamp}"
                    
                    try:
                        zip_data = create_download_zip(results, filename_prefix)
                        download_filename = f"{filename_prefix}_analysis.zip"
                        
                        download_link = get_download_link(zip_data, download_filename)
                        st.markdown(download_link, unsafe_allow_html=True)
                        st.success("✅ Analysis package ready for download!")
                        
                        st.info(f"""
                        📦 **Package Details:** **Filename:** {download_filename}  
                        **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
                        **Task:** {task.title()}  
                        **Model:** {selected_model_name}  
                        **Image:** {image_source}
                        """)
                    except Exception as e:
                        st.error(f"❌ Error creating download package: {e}")

def main():
    """Main application controller"""
    try:
        # Initialize session state
        if 'selected_task' not in st.session_state:
            st.session_state.selected_task = None
        
        # Route to appropriate interface
        if st.session_state.selected_task is None:
            show_landing_page()
        else:
            show_main_interface()
            
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.info("🔄 Please refresh the page if the issue persists")

if __name__ == "__main__":
    main()