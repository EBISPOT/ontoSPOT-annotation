import argparse
import chromadb
from openai import OpenAI
import numpy as np
import pandas as pd
import re
from oaklib import get_adapter
from transformers import AutoTokenizer, AutoModel
import torch

# ---------- Configuration ----------
CHROMA_PATH = "cl_embeddings"
COLLECTION_NAME = "cl_embeddings"
SUBSTRINGS_FILE = "substrings_to_remove.txt"
cl = get_adapter("sqlite:obo:cl")
# -----------------------------------

# Load substrings to remove
with open(SUBSTRINGS_FILE, 'r') as f:
    substrings_to_remove = [line.strip() for line in f.readlines()]

# Preprocessing function (shared)
def preprocess_label(label):
    label = label.lower()
    for substring in substrings_to_remove:
        label = label.replace(substring, '')
    return re.sub(r'\s+', ' ', label).strip()

# Cosine similarity
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# OpenAI embedding function
def openai_embed_batch(texts, model_name):
    oai_client = OpenAI()
    response = oai_client.embeddings.create(
        model=model_name,
        input=texts
    )
    return [embedding.embedding for embedding in response.data]

# BioBERT-like embedding function (sentence average of token embeddings)
def hf_embed_batch(texts, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    with torch.no_grad():
        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            # Mean of the last hidden state across tokens
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(emb.tolist())
    return embeddings

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map input cell type labels to CL using embedding similarity.")
    parser.add_argument("input_csv", help="Input CSV file with a column named 'input_label'")
    parser.add_argument("output_csv", help="Path to save the mapping output")
    parser.add_argument("--model", choices=["openai", "pubmedbert", "biobert"], default="openai", help="Embedding model to use")
    parser.add_argument("--openai_model_name", default="text-embedding-3-large", help="OpenAI model name")
    args = parser.parse_args()

    # Load input labels
    df = pd.read_csv(args.input_csv)
    input_labels = df['input_label'].tolist()
    preprocessed_labels = [preprocess_label(label) for label in input_labels]

    # Generate embeddings
    print(f"Embedding using {args.model}...")
    if args.model == "openai":
        input_embeddings = openai_embed_batch(preprocessed_labels, args.openai_model_name)
    elif args.model == "pubmedbert":
        input_embeddings = hf_embed_batch(preprocessed_labels, "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    elif args.model == "biobert":
        input_embeddings = hf_embed_batch(preprocessed_labels, "dmis-lab/biobert-base-cased-v1.1")

    # Load CL embeddings
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    all_records = collection.get(include=["embeddings", "documents"])
    cl_embeddings = np.array(all_records["embeddings"])
    cl_ids = all_records["ids"]

    # Match inputs
    results = []
    for label, emb in zip(input_labels, input_embeddings):
        distances = np.array([cosine_distance(emb, cl_emb) for cl_emb in cl_embeddings])
        best_idx = np.argmin(distances)
        best_id = cl_ids[best_idx].split("-")[0]
        best_label = cl.label(best_id)
        results.append({
            "input_label": label,
            "matched_label": best_label,
            "cl_id": best_id,
            "similarity_score": round(float(distances[best_idx]), 4)
        })
        print(f"Matched '{label}' → '{best_label}' ({best_id})")

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"\n✅ Done! Results saved to {args.output_csv}")