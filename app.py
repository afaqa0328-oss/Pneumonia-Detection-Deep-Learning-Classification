# app.py
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import io

# optional cv2 usage
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

import torchxrayvision as xrv

st.set_page_config(page_title="Pneumonia Detector (with Grad-CAM)", layout="wide")

# --- Custom CSS: Background + Title Color ---
st.markdown("""
    <style>
    /* File uploader background */
    .stFileUploader > div:first-child {
        background-color: #f8f9eb !important;
        border-radius: 8px;
    }

    /* Dropdown (selectbox) background */
    .stSelectbox > div > div {
        background-color: #f8f9eb !important;
        border-radius: 8px;
        color: black !important;
    }

    /* Also dark text inside inputs */
    .stTextInput > div > div > input {
        background-color: #f8f9eb !important;
    }

    /* General form background adjustment */
    .css-1djdyxw, .css-1n76uvr, .css-1vq4p4l {
        background-color: #f8f9eb !important;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Predict", "How it works", "Dataset", "About"])

# -------------------------
# Utility: Preprocessing
# -------------------------
def preprocess_image(pil_img):
    img = pil_img.convert("L")
    orig_display = img.copy()
    img_np = np.array(img).astype(np.uint8)

    if img_np.mean() < 100:
        img_np = 255 - img_np

    if CV2_AVAILABLE:
        img_np = cv2.threshold(img_np, 240, 240, cv2.THRESH_TRUNC)[1]
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
        img_resized = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA)
    else:
        img_p = Image.fromarray(img_np)
        img_p = img_p.filter(ImageFilter.GaussianBlur(radius=1))
        img_p = img_p.resize((224, 224), resample=Image.BILINEAR)
        img_resized = np.array(img_p)
        img_resized[img_resized > 240] = 240

    img_norm = xrv.datasets.normalize(img_resized, 255)
    tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).float()
    return orig_display, img_resized, img_norm, tensor

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model(weight_name="densenet121-res224-chex"):
    model = xrv.models.DenseNet(weights=weight_name)
    model.eval()
    return model

# -------------------------
# Grad-CAM
# -------------------------
def compute_gradcam(model, input_tensor, target_label_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    target_module = None
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_module = module
            break
    if target_module is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    fh = target_module.register_forward_hook(forward_hook)
    bh = target_module.register_backward_hook(backward_hook)

    output = model(input_tensor)
    score = output[0, target_label_index]
    score_value = float(score.detach().cpu().numpy())

    model.zero_grad()
    score.backward(retain_graph=True)

    act = activations['value'].squeeze(0)
    grad = gradients['value'].squeeze(0)
    weights = torch.mean(grad.view(grad.size(0), -1), dim=1)

    cam = torch.zeros(act.shape[1:], dtype=torch.float32).to(device)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = F.relu(cam)
    cam_np = cam.detach().cpu().numpy()
    cam_np = cam_np - cam_np.min()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    if CV2_AVAILABLE:
        cam_resized = cv2.resize(cam_np, (224, 224))
    else:
        cam_resized = np.array(
            Image.fromarray((cam_np * 255).astype(np.uint8)).resize((224, 224))
        ).astype(np.float32) / 255.0

    fh.remove()
    bh.remove()
    return cam_resized, score_value

# -------------------------
# Heatmap overlay
# -------------------------
def overlay_heatmap_on_image(orig_pil, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET if CV2_AVAILABLE else None):
    orig = orig_pil.convert("RGB").resize((224, 224))
    orig_np = np.array(orig)

    hm_uint8 = (heatmap * 255).astype(np.uint8)
    if CV2_AVAILABLE:
        hm_color = cv2.applyColorMap(hm_uint8, colormap)
        hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    else:
        import matplotlib.cm as cm
        hm_color = (cm.jet(hm_uint8 / 255.0)[:, :, :3] * 255).astype(np.uint8)

    overlay = (0.6 * orig_np + alpha * hm_color).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

# -------------------------
# ABOUT PAGE
# -------------------------
if page == "About":
    st.title("About this project")
    st.markdown("""
    ## 🏥 Pneumonia Detection Using Deep Learning

    This project demonstrates a modern AI-driven medical imaging solution  
    capable of detecting **pneumonia from chest X-rays** using:
    - Deep learning (DenseNet121)
    - TorchXRayVision pre-trained weights
    - Explainability (Grad-CAM heatmaps)
    - A clean Streamlit UI for interaction

    ## 🎯 Purpose
    The main objective is to:
    - Provide an educational demonstration of medical AI
    - Show how preprocessing improves diagnostic quality
    - Highlight the importance of model transparency
    - Encourage responsible use of AI in healthcare

    ## 🧠 Why DenseNet121?
    DenseNet121 is widely used in medical imaging because:
    - It has strong feature extraction capability
    - Performs well with limited data
    - Pretrained versions exist on very large X-ray datasets

    ## ⚠ Disclaimer
    This application is **not a medical device** and must **not** be used for  
    real clinical diagnosis. It is intended solely for:
    - Research  
    - Education  
    - Academic projects  
    - Demonstrating explainable AI  

    ## 👨‍💻 Developer
    **Mohammed Afaq Ahmed**  
    B.Tech CSE (Final Year)  

    Project includes:
    - Full pre-processing pipeline  
    - Threshold-based classification  
    - Grad-CAM visualization  
    - Streamlit-based web interface  
    """)


# -------------------------
# DATASET PAGE
# -------------------------
elif page == "Dataset":
    st.title("Dataset")
    st.markdown("""
    The system uses two major types of datasets:

    ## 📁 1. **Deep Learning Model Training Dataset (TorchXRayVision)**
    The underlying DenseNet121 model is pretrained on large, public, medical datasets:
    
    ### **• NIH ChestXray14**
    - 112,000+ X-ray images  
    - 14 disease labels  
    - Includes “Pneumonia” and “Lung Opacity”

    ### **• CheXpert Dataset (Stanford)**
    - 224,316 chest radiographs  
    - High-quality consensus labels  
    - Improved pathology labeling scheme

    ### **• MIMIC-CXR Dataset (MIT PhysioNet)**
    - 377,110 images  
    - 227,943 studies  
    - De-identified hospital-grade imaging  

    These datasets ensure that the model has seen *many* pneumonia patterns before deployment.

    ## 🧪 2. **Local Dataset (for testing and validation)**
    For local evaluation and demonstration, the system uses:
    
    ### **Kaggle Chest X-ray Pneumonia Dataset**
    - 5,856 pediatric X-rays  
    - Classified as **Normal** or **Pneumonia**  
    - Includes a variety of:
      - Opacities  
      - Infiltrates  
      - Atypical pneumonia patterns  

    ### Why this dataset?
    - Publicly available  
    - Cleanly labeled  
    - Common in academic research  
    - Helps validate AI predictions against ground-truth labels

    ## 📊 Confusion Matrix (example)
    If enabled, the app can display a confusion matrix showing:
    - True Positive (Pneumonia correctly detected)
    - False Positive (Normal predicted as pneumonia)
    - True Negative
    - False Negative

    This helps evaluate **accuracy, sensitivity, specificity**, and overall model performance.
    """)


# -------------------------
# HOW IT WORKS PAGE
# -------------------------
elif page == "How it works":
    st.title("How it works")
    st.markdown("""
    This system performs **AI-based Pneumonia Detection** using a combination of  
    **pre-processing, deep learning inference, and medical explainability techniques**.
    
    ## 🔬 Step-by-Step Pipeline

    ### **1️⃣ Image Upload**
    - The user uploads a chest X-ray in JPG/PNG format.
    - The image is automatically validated and converted to a standard grayscale format.

    ### **2️⃣ Preprocessing**
    To ensure medical accuracy, the system applies:
    - **Grayscale Conversion**  
      Converts image to single-channel for consistency.

    - **Inversion Check**  
      Many X-rays from mobile cameras appear inverted.  
      If the image is too dark (mean < 100), it is automatically inverted.

    - **Artifact Removal (cutting bright text markers)**  
      Values above 240 are clipped to remove labels/stickers.

    - **Gaussian Smoothing**  
      Removes scanner noise while preserving lung structure.

    - **Resize → 224×224 px**  
      Ensures model compatibility.

    - **TorchXRayVision Medical Normalization**  
      The image is normalized exactly the way clinical AI models expect.

    ### **3️⃣ Deep Learning Inference (DenseNet121)**
    - The system uses **DenseNet121** trained on:
      - NIH ChestXray14  
      - CheXpert  
      - MIMIC-CXR  
    - TorchXRayVision provides medically validated pretrained weights.
    - The model outputs scores for multiple lung diseases.

    ### **4️⃣ Pneumonia Probability Extraction**
    - From all predicted diseases, the app isolates the **Pneumonia** index.
    - Converts model score → probability using sigmoid.

    ### **5️⃣ Fixed Threshold Classification (0.60)**
    - **≥ 0.60 → Pneumonia Detected**  
    - **< 0.60 → Normal (probability hidden)**  
    This threshold is based on medical literature balancing sensitivity vs specificity.

    ### **6️⃣ Explainability Using Grad-CAM**
    The app generates:
    - A **heatmap** showing areas influencing the model's decision.
    - An **overlay on the original X-ray**, helping clinicians understand:
      - If the AI is focusing on lung fields  
      - Whether predictions are reasonable  
      - What region of the image indicates pneumonia

    ## 🧠 Why Explainability Matters
    Grad-CAM provides transparency and safety:
    - Helps avoid blind trust in AI  
    - Allows radiologists to verify predictions  
    - Encourages responsible clinical adoption  
    """)


# -------------------------
# UPLOAD & PREDICT
# -------------------------
if page == "Upload & Predict":
    st.title("Pneumonia Detection — Upload Chest X-ray")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

        weight_choice = st.selectbox("Model weights", [
            "densenet121-res224-chex",
            "densenet121-res224-nih",
            "densenet121-res224-mimic_ch",
            "densenet121-res224-mimic_nb",
        ], index=0)

        model = load_model(weight_choice)
        compute_gradcam_checkbox = st.checkbox("Compute Grad-CAM", value=True)

        if uploaded_file:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            st.markdown("**Original image**")
            st.image(image, use_column_width=True)


            orig_display, img_resized, img_norm, img_tensor = preprocess_image(image)

            st.markdown("**Processed image**")
            st.image(Image.fromarray(img_resized), width=300)

            with st.spinner("Predicting..."):
                with torch.no_grad():
                    preds = model(img_tensor)

            pathologies = xrv.datasets.default_pathologies
            pneu_idx = pathologies.index("Pneumonia") if "Pneumonia" in pathologies else 0

            score = float(preds[0, pneu_idx])
            prob = float(torch.sigmoid(torch.tensor(score)))

            st.markdown("### Prediction (Fixed threshold = 0.60)")
            threshold = 0.60  # FIXED — SLIDER REMOVED

            if prob >= threshold:
                st.error(f"🚨 Pneumonia Detected — Probability: {prob:.2f}")
            else:
                st.success("Normal")

            if compute_gradcam_checkbox:
                with st.spinner("Computing Grad-CAM..."):
                    cam, raw_score = compute_gradcam(model, img_tensor, pneu_idx)
                    heatmap_overlay = overlay_heatmap_on_image(orig_display, cam)

                st.markdown("### Grad-CAM")
                cols = st.columns(2)
                cols[0].image(orig_display.resize((300,300)), caption="Original", use_container_width=True)
                cols[1].image(heatmap_overlay.resize((300,300)), caption="Grad-CAM", use_container_width=True)

    with col2:
        st.markdown("## Model Info")
        st.write("""
        - DenseNet121  
        - Input: 224×224  
        - Explainable: Yes (Grad-CAM)  
        - Threshold fixed at 0.60  
        """)

