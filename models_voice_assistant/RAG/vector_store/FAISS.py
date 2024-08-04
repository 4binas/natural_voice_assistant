import json
import faiss
import numpy as np

from models_voice_assistant.RAG.encoder.gteLargeEn import GTELargeEncoder

class FAISSStore():
    def __init__(self, embeddingModel: GTELargeEncoder):
        self.index = faiss.IndexFlatL2(embeddingModel.embedding_length)
        self.embeddingModel = embeddingModel

        self.values = []

    def add_vector(self, text: str):
        self.add_vectors([text])

    def add_vectors(self, texts: list[list[float]]):
        vectors = []
        for i in range(0, len(texts), 10):
            embeddings = self.embeddingModel.batch_encode(texts[i:i+10])
            vectors = vectors + embeddings
        vectors = np.array(vectors)
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.values.extend(texts)
    
    def search(self, text: str, k: int) -> list[str]:
        vector = self.embeddingModel.encode(text)
        vector = np.array([vector])
        vector = vector.astype('float32')
        faiss.normalize_L2(vector)
        D, I = self.index.search(vector, k)
        res = []
        for i in range(k):
            res.append(self.values[I[0][i]])
        return res

    def write_index(self, path: str):
        faiss.write_index(self.index, path)
        json.dump(self.values, open(f"{path}.docs", "w"))
    
    def read_index(self, path: str):
        self.index = faiss.read_index(path)
        self.values = json.load(open(f"{path}.docs", "r"))