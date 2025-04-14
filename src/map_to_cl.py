import chromadb
from openai import OpenAI
import numpy as np
import pandas as pd
import re
from oaklib import get_adapter

# ---------- Configuration ----------
CHROMA_PATH = "cl_embeddings"
COLLECTION_NAME = "cl_embeddings"
INPUT_CSV = "input_data/cell_types.csv"  # One column: 'input_label'
OUTPUT_CSV = "output_data/cell_types-mapped.csv"
MODEL_NAME = "text-embedding-3-large"
SUBSTRINGS_FILE = "substrings_to_remove.txt"
cl = get_adapter("sqlite:obo:cl")
# -----------------------------------

# Load substrings to remove
with open(SUBSTRINGS_FILE, 'r') as f:
    substrings_to_remove = [line.strip() for line in f.readlines()]

# Preprocessing function (same as embed_cl.py)
def preprocess_label(label):
    label = label.lower()
    for substring in substrings_to_remove:
        label = label.replace(substring, '')
    return re.sub(r'\s+', ' ', label).strip()

# Load input data
df = pd.read_csv(INPUT_CSV)
input_labels = df['input_label'].tolist()
preprocessed_labels = [preprocess_label(label) for label in input_labels]

# Embed the input labels
oai_client = OpenAI()
def embed_batch(texts):
    response = oai_client.embeddings.create(
        model=MODEL_NAME,
        input=texts
    )
    return [embedding.embedding for embedding in response.data]

print("Embedding input cell types...")
input_embeddings = embed_batch(preprocessed_labels)

# Load the CL ChromaDB collection
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

# Get all existing embeddings and their IDs
print("Loading CL embeddings from ChromaDB...")
all_records = collection.get(include=["embeddings", "documents"])
cl_embeddings = np.array(all_records["embeddings"])
cl_ids = all_records["ids"]
cl_docs = all_records["documents"]

# Utility: Cosine distance
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Match each input to the closest CL term
results = []
for idx, (label, emb) in enumerate(zip(input_labels, input_embeddings)):
    distances = np.array([cosine_distance(emb, cl_emb) for cl_emb in cl_embeddings])
    best_idx = np.argmin(distances)
    best_score = distances[best_idx]
    best_id = cl_ids[best_idx].split("-")[0]  # Strip label/synonym suffix
    best_match = cl.label(best_id)  # Always get the preferred label from OAKlib
    results.append({
        "input_label": label,
        "matched_label": best_match,
        "cl_id": best_id,
        "similarity_score": round(float(best_score), 4)
    })
    print(f"Matched '{label}' → '{best_match}' ({best_id}) [score: {best_score:.4f}]")

# Save results
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Done! Output written to {OUTPUT_CSV}")
