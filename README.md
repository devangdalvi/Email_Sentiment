🧠📩 Email → GNN Sentiment Analysis System
An end-to-end Email Sentiment Analysis System that leverages Natural Language Processing (NLP) and Graph Neural Networks (GNNs) to classify incoming emails as Positive, Negative, or Neutral in real time.

🚀 Overview

This project automatically reads incoming emails from a Gmail inbox, processes the content, and predicts sentiment using a hybrid AI pipeline:

🔤 Sentence Transformers for text embeddings
🕸️ Graph Neural Network (GraphSAGE) for relational learning
📊 Real-time sentiment prediction

🏗️ High-Level Architecture
<img width="690" height="452" alt="image" src="https://github.com/user-attachments/assets/7f943b22-886e-4940-b349-70a95b0a56eb" />

🔬 Component-Wise Architecture
📩 1. Email Ingestion Layer
Uses IMAP protocol
Fetches unread emails from Gmail
Extracts:
Subject
Body

📁 File: app.py

🧹 2. Preprocessing Layer

Combines subject + body
Cleans and decodes text
Prepares input for embedding

🔤 3. Embedding Layer

Model: all-mpnet-base-v2
Converts text → 768-dimensional vector

📁 File: inference.py

🕸️ 4. Graph Construction (Pre-trained)

Built during training phase
Structure:
Nodes = Sentences
Edges = Cosine similarity (KNN-based)

📦 Stored as:

sentiment_graph_data.pkl
Contains:

Node features (embeddings)
edge_index
Labels

🧠 5. GNN Model Layer

Model: GraphSAGE (3-layer architecture)
Input (768)
   ↓
SAGEConv → ReLU → Dropout
   ↓
SAGEConv → ReLU → Dropout
   ↓
SAGEConv → ReLU
   ↓
Fully Connected Layer
   ↓
Output (3 classes)

🎯 6. Prediction Logic

Steps:

Convert email → embedding
Compare with graph nodes (cosine similarity)
Find closest node
Pass through GNN
Generate probabilities
Select highest class

📊 7. Output Layer

Example:

📨 Email Content: "Your service is amazing!"
🧠 Prediction:
Sentiment: Positive
Confidence: 92.3%


🏗️ Project Structure

├── app.py                  # Email Reader (IMAP)
├── inference.py            # Embedding + Prediction Logic
├── model.py                # GraphSAGE Architecture
├── models/
│   ├── sentiment_gnn_model.pth
│   └── sentiment_graph_data.pkl
├── requirements.txt
└── README.md

👨‍💻 Author

Devang Dalvi
