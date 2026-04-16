import torch
import pickle
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from model import SentimentGraphSAGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/sentiment_gnn_model.pth"
GRAPH_PATH = "models/sentiment_graph_data.pkl"

# Load graph
with open(GRAPH_PATH, "rb") as f:
    graph_data = pickle.load(f)

print("✅ Graph loaded")

# Load model
model = SentimentGraphSAGE(
    in_channels=graph_data.num_features,
    num_classes=graph_data.num_classes
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded")

# Load embedding model
embed_model = SentenceTransformer("all-mpnet-base-v2")
# embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def predict_sentiment(text):
    # Convert text → embedding
    emb = embed_model.encode(text)

    node_embeddings = graph_data.x.cpu().numpy()

    # Cosine similarity
    sim = np.dot(node_embeddings, emb) / (
        np.linalg.norm(node_embeddings, axis=1) * np.linalg.norm(emb) + 1e-8
    )

    idx = np.argmax(sim)

    with torch.no_grad():
        out = model(graph_data.x.to(device), graph_data.edge_index.to(device))
        probs = F.softmax(out[idx], dim=0).cpu().numpy()

    pred = np.argmax(probs)
    label = graph_data.class_names[pred]
    confidence = probs[pred] * 100

    return label, confidence