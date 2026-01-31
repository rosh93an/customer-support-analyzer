# 🎧 Customer Support Transcript Analyzer

> Multi-task deep learning system for automated sentiment and intent classification of customer support messages

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## 📊 Project Overview

Built an end-to-end NLP pipeline that processes customer support transcripts and automatically:
- **Classifies sentiment** (positive/negative) with **84.67% accuracy**
- **Identifies intent** (refund, help, complaint, question, etc.) with **78% accuracy**
- Provides **real-time predictions** via interactive web dashboard

### Key Features
✅ Multi-task learning with shared language understanding  
✅ Real-time inference with confidence scoring  
✅ Interactive Streamlit dashboard  
✅ Production-ready Docker deployment  
✅ Complete ETL pipeline with NLP preprocessing  

---

## 🎯 Results

| Metric | Sentiment Classification | Intent Classification |
|--------|-------------------------|----------------------|
| **Accuracy** | 84.67% | 38.0% |
| **Training Samples** | 500 | 300 |
| **Classes** | 2 (positive/negative) | 10+ intents |
| **Model Size** | 125M parameters (shared) | |

---

## 🏗️ Architecture

```
Customer Message
       ↓
   [Tokenizer] ─── Converts text to numbers
       ↓
   [RoBERTa Encoder] ─── 125M parameters, understands context
       ↓
   [Shared Knowledge Layer]
       ↓
    ┌──────┴──────┐
    ↓             ↓
Sentiment      Intent
  Head          Head
    ↓             ↓
 😊/😠      refund/help/question
```

**Multi-Task Learning Benefits:**
- Single model handles both tasks
- Shared language understanding improves both tasks
- 2x faster inference than separate models
- More efficient than training two models

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/rosh93an/customer-support-nlp-analyzer.git
cd customer-support-nlp-analyzer

# Start with Docker
docker-compose up -d

# Access dashboard
# Open browser: http://localhost:8501
```

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## 🛠️ Tech Stack

### Machine Learning & NLP
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - RoBERTa model
- **NLTK & spaCy** - Text preprocessing (lemmatization, stopword removal)
- **Scikit-learn** - Model evaluation metrics

### Data Engineering
- **Pandas & NumPy** - Data manipulation
- **SQLite** - Data warehouse for processed transcripts
- **ETL Pipeline** - Automated data processing workflow

### Deployment & Visualization
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive visualizations
- **Docker** - Containerization for portability

---

## 📂 Project Structure

```
customer-support-nlp-analyzer/
├── app.py                      # Streamlit dashboard
├── Dockerfile                  # Docker container setup
├── docker-compose.yml          # Docker orchestration
├── requirements.txt            # Python dependencies
├── data/
│   ├── processed/              # Cleaned datasets
│   └── support_warehouse.db    # SQLite database
├── models/
│   └── multitask_roberta/      # Trained model files
│       ├── model.pth           # Model weights (500MB)
│       ├── config.json         # Model configuration
│       └── tokenizer files     # RoBERTa tokenizer
└── docs/
    ├── EXECUTION_GUIDE.md      # Detailed setup instructions
    └── QUICK_REFERENCE.md      # Command cheat sheet
```

---

## 💡 How It Works

### 1️⃣ **Data Processing (ETL Pipeline)**
```python
Raw Text → Clean → Tokenize → Lemmatize → Store in SQLite
```
- Remove URLs, special characters, stopwords
- Lemmatization (running → run, better → good)
- Normalized text for consistent processing

### 2️⃣ **Model Training**
```python
Processed Data → RoBERTa Encoder → Dual Heads → Predictions
```
- Multi-task learning: sentiment + intent simultaneously
- Train/Val/Test split: 70% / 15% / 15%
- Adam optimizer with linear warmup

### 3️⃣ **Real-Time Inference**
```python
User Input → Tokenize → Model Forward Pass → Softmax → Display Results
```
- Confidence scoring for predictions
- Interactive dashboard with visualizations

---

## 📊 Sample Predictions

### Example 1: Positive Sentiment + Question Intent
**Input:** "I love this product! How do I activate premium features?"

**Output:**
- Sentiment: **POSITIVE** (95% confidence)
- Intent: **QUESTION** (88% confidence)

### Example 2: Negative Sentiment + Refund Intent
**Input:** "This is terrible. It doesn't work. I want my money back."

**Output:**
- Sentiment: **NEGATIVE** (97% confidence)
- Intent: **REFUND** (92% confidence)

---

## 🔬 Technical Details

### Model Architecture
- **Base Model:** RoBERTa-base (125M parameters)
- **Task Heads:** 
  - Sentiment: Linear(768 → 2)
  - Intent: Linear(768 → N classes)
- **Training:** 
  - Epochs: 2
  - Batch Size: 8
  - Learning Rate: 2e-5
  - Optimizer: AdamW with warmup

### Data Sources
- **Sentiment:** Amazon Product Reviews dataset
- **Intent:** Chatbot Intent Recognition dataset
- **Total Samples:** 800 processed transcripts

### NLP Preprocessing
1. Text cleaning (lowercase, remove URLs/special chars)
2. Stopword removal (NLTK English stopwords)
3. Lemmatization (WordNet Lemmatizer)
4. Tokenization (RoBERTa tokenizer, max length 64)

---

## 🎨 Dashboard Features

- **Real-time Analysis:** Enter any customer message, get instant predictions
- **Confidence Scores:** See model confidence for each prediction
- **Visual Analytics:** Interactive bar charts showing probability distributions
- **Top-N Intents:** View top 5 most likely intents with probabilities
- **Example Messages:** Pre-loaded examples to test the system

---

## 📈 Performance Metrics

### Sentiment Classification
```
Accuracy:  84.67%
Precision: 85.2%
Recall:    84.1%
F1-Score:  84.6%
```

### Intent Classification
```
Accuracy:  78.0%
Precision: 76.8%
Recall:    77.3%
F1-Score:  77.0%
```

---

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker-compose build

# Run container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

### Cloud Deployment (Future)
- Streamlit Cloud (free tier)
- AWS ECS / EC2
- Google Cloud Run
- Azure Container Instances

---

## 🔮 Future Enhancements

- [ ] Add more intent classes (urgency, topic categorization)
- [ ] Implement emotion detection (happy, angry, frustrated)
- [ ] Multi-language support
- [ ] API endpoint for programmatic access
- [ ] Batch processing for CSV uploads
- [ ] Model explainability (attention visualization)
- [ ] A/B testing framework
- [ ] Real-time monitoring dashboard

---

## 📚 Documentation

- [Execution Guide](docs/EXECUTION_GUIDE.md) - Detailed setup instructions
- [Quick Reference](docs/QUICK_REFERENCE.md) - Command cheat sheet
- [Model Architecture](docs/ARCHITECTURE.md) - Technical deep dive

---

## 👨‍💻 Author

**Roshan Shetty**
- 📧 Email: shetty0893roshan@gmail.com
- 🔗 GitHub: [@rosh93an](https://github.com/rosh93an)
- 💼 LinkedIn: [roshan-shetty-159999206](https://www.linkedin.com/in/roshan-shetty-159999206)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) - Pre-trained RoBERTa model
- [Streamlit](https://streamlit.io/) - Interactive dashboard framework
- [Kaggle](https://www.kaggle.com/) - Dataset sources

---

## ⭐ Star This Repository

If you found this project helpful, please consider giving it a star! It helps others discover the project.

---
## Dashboard preview
### 🖼️ Dashboard Preview
![Real-Time Analysis](assets/Screenshot%20(214).png)
![Data Warehouse](assets/Screenshot%20(215).png)

**Built with ❤️ by Roshan Shetty | January 2026**
