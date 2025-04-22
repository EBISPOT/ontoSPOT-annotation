# embedding_backends.py

def embed_openai(texts, model_name="text-embedding-3-large"):
    from openai import OpenAI
    client = OpenAI()
    response = client.embeddings.create(model=model_name, input=texts)
    return [embedding.embedding for embedding in response.data]

def embed_pubmedbert(texts):
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).numpy().tolist()

def embed_biobert(texts):
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).numpy().tolist()
