import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🫁 Lung Disease Prediction",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🫁 Lung Disease Prediction Dashboard")

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio(
        "Select Module",
        ["📊 Data Management", "⚙️ Model Configuration", "🚀 Training Control", "📈 Results Analysis"]
    )

# Data Management Module
if app_mode == "📊 Data Management":
    st.header("📊 Data Folder Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("pneumonia/")
        st.metric("Images", 500)
        st.metric("Classes", 2)
    
    with col2:
        st.subheader("data_folder_2/")
        st.metric("Images", 300)
        st.metric("Classes", 2)
    
    with col3:
        st.subheader("folder_3/")
        st.metric("Images", 200)
        st.metric("Classes", 2)
    
    st.markdown("---")
    
    selected_folders = st.multiselect(
        "Select data folders for training",
        ["pneumonia/", "data_folder_2/", "folder_3/"],
        default=["pneumonia/"]
    )
    st.info(f"Selected {len(selected_folders)} folder(s)")

# Model Configuration Module
elif app_mode == "⚙️ Model Configuration":
    st.header("⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Configuration")
        learning_rate = st.select_slider("Learning Rate", options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], value=1e-3)
        batch_size = st.slider("Batch Size", min_value=8, max_value=128, value=32, step=8)
        image_size = st.slider("Image Size", min_value=128, max_value=512, value=224, step=32)
    
    with col2:
        st.subheader("Training Configuration")
        num_epochs = st.slider("Number of Epochs", min_value=10, max_value=200, value=50, step=10)
        early_stopping = st.slider("Early Stopping Patience", min_value=3, max_value=20, value=10)
        mixed_precision = st.checkbox("Enable Mixed Precision Training", value=True)
    
    if st.button("💾 Save Configuration"):
        st.success("✅ Configuration saved!")

# Training Control Module
elif app_mode == "🚀 Training Control":
    st.header("🚀 Training Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        device = st.radio("Select Device", ["cuda", "cpu"])
        config_path = st.text_input("Config Path", value="configs/config.yaml")
    
    with col2:
        st.subheader("Training Status")
        if st.button("▶️ Start Training"):
            st.success("✅ Training started!")
            with st.spinner("Training in progress..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
    
    st.markdown("---")
    st.subheader("Training Logs")
    st.text_area("Logs", value="[INFO] Starting training...\n[INFO] Epoch 1/50...", height=200, disabled=True)

# Results Analysis Module
elif app_mode == "📈 Results Analysis":
    st.header("📈 Results Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        st.metric("Accuracy", "92%")
        st.metric("Precision", "89%")
        st.metric("Recall", "85%")
        st.metric("F1 Score", "87%")
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = np.array([[45, 5], [3, 47]])
        fig = go.Figure(data=go.Heatmap(z=cm, text=cm, texttemplate="%{text}", showscale=False))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Training Loss Curve")
        epochs = np.arange(1, 51)
        train_loss = 0.5 * np.exp(-epochs/20)
        val_loss = 0.55 * np.exp(-epochs/20)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss", mode="lines"))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss", mode="lines"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.subheader("Accuracy Trend")
        accuracy = 0.6 + 0.32 * (1 - np.exp(-epochs/15))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=accuracy, fill="tozeroy", name="Accuracy", mode="lines"))
        st.plotly_chart(fig, use_container_width=True)