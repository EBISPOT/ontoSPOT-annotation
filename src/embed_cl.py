import chromadb
import os
from oaklib import get_adapter
import openai
from chromadb import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI
import re

# Read substrings to remove from file
substrings_to_remove = [line.strip('\n') for line in open('substrings_to_remove.txt').readlines()]

# Set up OpenAI client
oai_client = OpenAI()

def embed_batch(texts):
    response = oai_client.embeddings.create(
        model="text-embedding-3-large", 
        input=texts
    )
    return [embedding.embedding for embedding in response.data]

def preprocess_label(label):
    label = label.lower()
    for substring in substrings_to_remove:
        label = label.replace(substring, '')
    return re.sub(r'\s+', ' ', label).strip()

# Set up ChromaDB for CL
client = chromadb.PersistentClient(path="cl_embeddings")
collection = client.create_collection(name="cl_embeddings", get_or_create=True)

# Load CL ontology
cl = get_adapter("sqlite:obo:cl")

print('Building documents', flush=True)

docs = []
ids = []

for entity in cl.entities():
    if not entity.startswith("CL:"):
        continue
    # Optional: limit to subclasses of 'native cell' CL:0000003 or just check that it's under CL:0000000
    if not "CL:0000000" in cl.ancestors(entity):
        continue
    label = cl.label(entity)
    if not label:
        continue
    docs.append(preprocess_label(label))
    ids.append(entity + '-label')
    alias_map = cl.entity_alias_map(entity)
    if 'oio:hasExactSynonym' in alias_map:
        for idx, alias in enumerate(alias_map['oio:hasExactSynonym'], start=1):
            docs.append(preprocess_label(alias))
            ids.append(f"{entity}-synonym-{idx}")

print('Creating embeddings in batches', flush=True)

batch_size = 1000
n = 0
embeddings = []

for i in range(0, len(docs), batch_size):
    batch_docs = docs[i:i+batch_size]
    batch_embeddings = embed_batch(batch_docs)
    embeddings.extend(batch_embeddings)
    n += len(batch_docs)
    print(f"Embedded {n} of {len(docs)} ({n/len(docs)*100:.2f}%)", flush=True)

collection.add(
    documents=docs,
    ids=ids,
    embeddings=embeddings
)
