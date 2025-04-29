# Sentiment-Analysis-IndoBert

This project performs **sentiment analysis** on Indonesian restaurant reviews using the **IndoBERT** language model. It classifies each review as either **positive** or **negative**, helping businesses and researchers understand customer feedback at scale.

## 🧠 Project Overview

Sentiment analysis is a key application in Natural Language Processing (NLP) used to gauge public opinion and customer satisfaction. This project specifically focuses on:

- **Language**: Indonesian  
- **Domain**: Restaurant reviews  
- **Task**: Binary sentiment classification (Positive / Negative)

We leverage **IndoBERT**, a pre-trained BERT model for the Indonesian language, to achieve accurate and context-aware sentiment classification.

---

## 🧹 Preprocessing

1. Lowercased text
2. Compound phrases like "tidak enak" transformed into single tokens (`tidak_enak`)
3. Lemmatization using `spaCy` (with `en_core_web_sm`)
4. Custom stopwords for Indonesian slang and filler words (e.g., "nih", "deh", "dong")

## ⚖️ Balancing

Training data was balanced using **upsampling**:
- `Positive`: ~14,477
- `Negative`: ~14,477

Validation set remains imbalanced to test generalizability.

## 📊 Word Frequency & WordClouds

- Most frequent words in positive vs. negative reviews were visualized.
- Used `Tokenizer` and `WordCloud` to display insights.

## 🤗 IndoBERT Model (Fine-Tuned)

### 🔧 Model & Tokenizer

- Base: [`indolem/indobert-base-uncased`](https://huggingface.co/indolem/indobert-base-uncased)
- Output: Binary classification (Positive or Negative)
- Dropout increased for regularization

### 🔢 Tokenization

- Used `AutoTokenizer` from HuggingFace
- Max length: 128 tokens
- Tokenized reviews using `.apply(preprocess_text)` + IndoBERT's tokenizer

### 🧠 Model Training

- Framework: PyTorch
- Batch size: 16
- Optimizer: `AdamW` with learning rate `2e-5` and weight decay `0.01`
- Device: GPU (if available)
- Epochs: 4

### 📦 Dataloaders

- `TensorDataset` created for both training and validation
- Dataloader uses `RandomSampler` for shuffling

## 🖼️ WordCloud Examples

![image](https://github.com/user-attachments/assets/5f128c13-a8d1-41e4-838d-65a30ae8dd03)
Common tokens like `enak`, `makan`, `bagus`, `harga`, `tempat` dominate both classes, requiring careful modeling.
