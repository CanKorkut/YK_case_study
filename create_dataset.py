
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


def create_dataset():

    dataset = load_dataset('microsoft/wiki_qa', split='train[:1000]')

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    contexts = [item['answer'] for item in dataset]
    embeddings = model.encode(contexts)

    embedding_dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(embedding_dim)

    faiss.normalize_L2(embeddings)
    index.add(np.array(embeddings))

    return index, contexts, model