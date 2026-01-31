"""
Customer Support Transcript Analysis Dashboard
Multi-Task RoBERTa Model for Sentiment & Intent Classification
"""

import streamlit as st
import torch
import json
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
#from transformers import RobertaTokenizer
from transformers import AutoTokenizer
from torch import nn
from transformers import RobertaModel
import numpy as np

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="Support Transcript Analyzer",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# MODEL DEFINITION (Same as training)
# ===========================
class MultiTaskRoBERTa(nn.Module):
    def __init__(self, num_sentiment_classes, num_intent_classes, dropout=0.3):
        super(MultiTaskRoBERTa, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.sentiment_classifier = nn.Linear(self.roberta.config.hidden_size, num_sentiment_classes)
        self.intent_classifier = nn.Linear(self.roberta.config.hidden_size, num_intent_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        return sentiment_logits, intent_logits

# ===========================
# LOAD MODEL & RESOURCES
# ===========================
@st.cache_resource
def load_model():
    """Load trained model and tokenizer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load label encoders
    with open('models/label_encoders.json', 'r') as f:
        encoders = json.load(f)
    
    sentiment_classes = encoders['sentiment_classes']
    intent_classes = encoders['intent_classes']
    
    # Load model
    model = MultiTaskRoBERTa(
        num_sentiment_classes=len(sentiment_classes),
        num_intent_classes=len(intent_classes)
    )
    
    checkpoint = torch.load('models/multitask_roberta/model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    return model, tokenizer, sentiment_classes, intent_classes, device, checkpoint['history']

@st.cache_data
def load_warehouse_data():
    """Load data from SQLite warehouse"""
    conn = sqlite3.connect('data/support_warehouse.db')
    
    sentiment_df = pd.read_sql('SELECT * FROM sentiment_data', conn)
    intent_df = pd.read_sql('SELECT * FROM intent_data', conn)
    metadata = pd.read_sql('SELECT * FROM metadata', conn)
    
    conn.close()
    
    return sentiment_df, intent_df, metadata

# ===========================
# PREDICTION FUNCTION
# ===========================
def predict_text(text, model, tokenizer, sentiment_classes, intent_classes, device):
    """Predict sentiment and intent for input text"""
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        sentiment_logits, intent_logits = model(input_ids, attention_mask)
    
    # Get predictions
    sentiment_probs = torch.softmax(sentiment_logits, dim=1)[0].cpu().numpy()
    intent_probs = torch.softmax(intent_logits, dim=1)[0].cpu().numpy()
    
    sentiment_pred = sentiment_classes[np.argmax(sentiment_probs)]
    intent_pred = intent_classes[np.argmax(intent_probs)]
    
    sentiment_confidence = float(np.max(sentiment_probs))
    intent_confidence = float(np.max(intent_probs))
    
    return {
        'sentiment': sentiment_pred,
        'sentiment_confidence': sentiment_confidence,
        'sentiment_probs': dict(zip(sentiment_classes, sentiment_probs)),
        'intent': intent_pred,
        'intent_confidence': intent_confidence,
        'intent_probs': dict(zip(intent_classes, intent_probs))
    }

# ===========================
# MAIN APP
# ===========================
def main():
    # Header
    st.title("🎧 Customer Support Transcript Analyzer")
    st.markdown("### Multi-Task RoBERTa Model for Sentiment & Intent Classification")
    st.markdown("---")
    
    # Load resources
    with st.spinner("🔄 Loading model and data..."):
        model, tokenizer, sentiment_classes, intent_classes, device, history = load_model()
        sentiment_df, intent_df, metadata = load_warehouse_data()
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🔍 Real-Time Analysis", "📈 Model Performance", "🗄️ Data Warehouse", "ℹ️ About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Model Info**\n"
        f"- Architecture: Multi-Task RoBERTa\n"
        f"- Sentiment Classes: {len(sentiment_classes)}\n"
        f"- Intent Classes: {len(intent_classes)}\n"
        f"- Device: {device.type.upper()}"
    )
    
    # ===========================
    # PAGE 1: REAL-TIME ANALYSIS
    # ===========================
    if page == "🔍 Real-Time Analysis":
        st.header("🔍 Analyze Customer Transcript")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input text
            user_input = st.text_area(
                "Enter customer support transcript:",
                height=150,
                placeholder="Type or paste a customer message here..."
            )
            
            analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
        
        with col2:
            st.info(
                "**Tips:**\n"
                "- Enter actual customer messages\n"
                "- Works best with 10-200 words\n"
                "- Supports informal language\n"
                "- Real-time predictions"
            )
        
        # Prediction
        if analyze_btn and user_input:
            with st.spinner("🤖 Analyzing..."):
                results = predict_text(
                    user_input, model, tokenizer, 
                    sentiment_classes, intent_classes, device
                )
            
            st.success("✅ Analysis Complete!")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("😊 Sentiment Analysis")
                
                # Sentiment badge
                sentiment_color = "green" if results['sentiment'] == 'positive' else "red"
                st.markdown(
                    f"<h2 style='text-align: center; color: {sentiment_color};'>"
                    f"{results['sentiment'].upper()}</h2>",
                    unsafe_allow_html=True
                )
                st.metric("Confidence", f"{results['sentiment_confidence']:.2%}")
                
                # Probability chart
                sent_df = pd.DataFrame([
                    {"Sentiment": k, "Probability": v} 
                    for k, v in results['sentiment_probs'].items()
                ])
                fig = px.bar(sent_df, x="Sentiment", y="Probability", 
                           title="Sentiment Probabilities")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Intent Classification")
                
                # Intent badge
                st.markdown(
                    f"<h2 style='text-align: center; color: blue;'>"
                    f"{results['intent'].upper()}</h2>",
                    unsafe_allow_html=True
                )
                st.metric("Confidence", f"{results['intent_confidence']:.2%}")
                
                # Top 5 intents
                intent_probs = sorted(results['intent_probs'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                intent_df = pd.DataFrame(intent_probs, columns=["Intent", "Probability"])
                fig = px.bar(intent_df, x="Probability", y="Intent", 
                           orientation='h', title="Top 5 Intent Predictions")
                st.plotly_chart(fig, use_container_width=True)
            
            # Raw predictions
            with st.expander("🔬 View Raw Predictions"):
                st.json(results)
    
    # ===========================
    # PAGE 2: MODEL PERFORMANCE
    # ===========================
    elif page == "📈 Model Performance":
        st.header("📈 Model Training Metrics")
        
        # Training history charts
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], 
                                   mode='lines+markers', name='Train Loss'))
            fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], 
                                   mode='lines+markers', name='Val Loss'))
            fig.update_layout(title='Training & Validation Loss', 
                            xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment accuracy
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=history['train_sentiment_acc'], 
                                   mode='lines+markers', name='Train Acc'))
            fig.add_trace(go.Scatter(x=epochs, y=history['val_sentiment_acc'], 
                                   mode='lines+markers', name='Val Acc'))
            fig.update_layout(title='Sentiment Classification Accuracy', 
                            xaxis_title='Epoch', yaxis_title='Accuracy')
            st.plotly_chart(fig, use_container_width=True)
        
        # Intent accuracy
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=history['train_intent_acc'], 
                               mode='lines+markers', name='Train Acc'))
        fig.add_trace(go.Scatter(x=epochs, y=history['val_intent_acc'], 
                               mode='lines+markers', name='Val Acc'))
        fig.update_layout(title='Intent Classification Accuracy', 
                        xaxis_title='Epoch', yaxis_title='Accuracy')
        st.plotly_chart(fig, use_container_width=True)
        
        # Final metrics
        st.subheader("🎯 Final Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
        col2.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
        col3.metric("Sentiment Acc", f"{history['val_sentiment_acc'][-1]:.2%}")
        col4.metric("Intent Acc", f"{history['val_intent_acc'][-1]:.2%}")
    
    # ===========================
    # PAGE 3: DATA WAREHOUSE
    # ===========================
    elif page == "🗄️ Data Warehouse":
        st.header("🗄️ SQLite Data Warehouse")
        
        # Metadata
        st.subheader("📊 Database Metadata")
        st.dataframe(metadata, use_container_width=True)
        
        # Tabs for different tables
        tab1, tab2 = st.tabs(["Sentiment Data", "Intent Data"])
        
        with tab1:
            st.subheader("😊 Sentiment Dataset")
            st.dataframe(sentiment_df.head(100), use_container_width=True)
            
            # Distribution
            fig = px.histogram(sentiment_df, x='sentiment', 
                             title='Sentiment Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🎯 Intent Dataset")
            st.dataframe(intent_df.head(100), use_container_width=True)
            
            # Distribution
            intent_counts = intent_df['intent'].value_counts().head(10)
            fig = px.bar(x=intent_counts.index, y=intent_counts.values,
                       title='Top 10 Intents Distribution',
                       labels={'x': 'Intent', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
    
    # ===========================
    # PAGE 4: ABOUT
    # ===========================
    else:
        st.header("ℹ️ About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        This is a complete **ETL Pipeline** with a **Multi-Task Deep Learning Model** 
        for analyzing customer support transcripts.
        
        ### 🔧 Architecture
        
        **1. Data Sources**
        - Amazon Reviews (Sentiment Analysis)
        - Chatbot Intent Recognition Dataset
        
        **2. ETL Pipeline**
        - **Extract**: Download data from Kaggle via kagglehub
        - **Transform**: NLP preprocessing (cleaning, stopword removal, lemmatization)
        - **Load**: Store in SQLite database
        
        **3. Model Architecture**
        - Base: RoBERTa (Robustly Optimized BERT)
        - Multi-Task Learning with 2 heads:
          - Sentiment Classification (2 classes)
          - Intent Classification (N classes)
        - Shared encoder, separate task-specific heads
        
        **4. Deployment**
        - Streamlit dashboard for real-time inference
        - Docker containerization for portability
        
        ### 📊 Key Features
        - ✅ Real-time sentiment & intent prediction
        - ✅ Confidence scores & probability distributions
        - ✅ Training metrics visualization
        - ✅ Data warehouse exploration
        - ✅ Containerized deployment
        
        ### 🚀 Tech Stack
        - **Data**: Kaggle, pandas, SQLite
        - **NLP**: NLTK, spaCy, transformers
        - **ML**: PyTorch, Hugging Face Transformers
        - **Visualization**: Plotly, Streamlit
        - **Deployment**: Docker
        
        ### 📝 Usage Instructions
        1. Navigate to "Real-Time Analysis" page
        2. Enter customer message
        3. Click "Analyze" button
        4. View sentiment & intent predictions
        
        ### 👨‍💻 Project Structure
        ```
        project/
        ├── 1_data_download.ipynb
        ├── 2_etl_pipeline.ipynb
        ├── 3_model_training.ipynb
        ├── 4_streamlit_app.py
        ├── Dockerfile
        ├── data/
        │   ├── processed/
        │   └── support_warehouse.db
        └── models/
            └── multitask_roberta/
        ```
        """)

if __name__ == "__main__":
    main()