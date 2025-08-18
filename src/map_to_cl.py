import chromadb
import argparse
import logging
import numpy as np
import pandas as pd
import re
from openai import OpenAI
from oaklib import get_adapter

# ---------- Configuration ----------
CHROMA_PATH = "cl_embeddings"
COLLECTION_NAME = "cl_embeddings"
SUBSTRINGS_FILE = "substrings_to_remove.txt"
MODEL_NAME = "text-embedding-3-large"
cl = get_adapter("sqlite:obo:cl")
# -----------------------------------

# Load substrings to remove
with open(SUBSTRINGS_FILE, 'r') as f:
    substrings_to_remove = [line.strip() for line in f.readlines()]

# Preprocessing function
def preprocess_label(label):
    label = label.lower()
    for substring in substrings_to_remove:
        label = label.replace(substring, '')
    return re.sub(r'\s+', ' ', label).strip()

# Embed text batch with OpenAI
def embed_batch(texts, model_name):
    response = OpenAI().embeddings.create(
        model=model_name,
        input=texts
    )
    return [embedding.embedding for embedding in response.data]

# Cosine distance
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Load input data
    logging.info("Loading input data...")
    df = pd.read_csv(args.input_csv)
    input_labels = df['input_label'].tolist()
    preprocessed_labels = [preprocess_label(label) for label in input_labels]

    # Deduplicate labels
    unique_labels = list(dict.fromkeys(preprocessed_labels))  # preserves order

    # Embed unique labels
    logging.info(f"Embedding {len(unique_labels)} unique input labels...")
    input_embeddings = []
    for i in range(0, len(unique_labels), args.batch_size):
        batch = unique_labels[i:i + args.batch_size]
        input_embeddings.extend(embed_batch(batch, args.model_name))
        logging.debug(f"Embedded {i + len(batch)} / {len(unique_labels)}")

    # Load CL embeddings from ChromaDB
    logging.info("Loading CL embeddings from ChromaDB...")
    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_collection(args.collection_name)
    all_records = collection.get(include=["embeddings", "documents", "ids"])
    cl_embeddings = np.array(all_records["embeddings"])
    cl_ids = all_records["ids"]

    # Map unique labels to closest CL term
    label_to_mapping = {}
    for label, emb in zip(unique_labels, input_embeddings):
        distances = np.array([cosine_distance(emb, cl_emb) for cl_emb in cl_embeddings])
        best_idx = np.argmin(distances)
        best_score = distances[best_idx]
        best_id = cl_ids[best_idx].split("-")[0]
        best_label = cl.label(best_id)

        label_to_mapping[label] = {
            "matched_label": best_label,
            "cl_id": best_id,
            "similarity_score": round(float(best_score), 4)
        }

        logging.debug(f"Mapped '{label}' → '{best_label}' ({best_id}) [score: {best_score:.4f}]")

    # Assign mapped results back to original labels
    results = []
    for label in preprocessed_labels:
        mapping = label_to_mapping[label]
        results.append({
            "input_label": label,
            "matched_label": mapping["matched_label"],
            "cl_id": mapping["cl_id"],
            "similarity_score": mapping["similarity_score"]
        })

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    logging.info(f"Output written to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map input cell types to CL using OpenAI embeddings and cosine similarity.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with 'input_label' column")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding input labels")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="OpenAI embedding model name")
    parser.add_argument("--chroma-path", type=str, default=CHROMA_PATH, help="Path to ChromaDB persistent directory")
    parser.add_argument("--collection-name", type=str, default=COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    main(args)
