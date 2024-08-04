from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class GTELargeEncoder:
  def __init__(self, model_name: str = 'Alibaba-NLP/gte-large-en-v1.5', device: str = 'cuda', embedding_length: int = 1024):
    self.device = device
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
    self.embedding_length = embedding_length

  def encode(self, text: str) -> list[float]:
    res = self.batch_encode([text])
    return res[0]

  def batch_encode(self, texts: list[str]) -> list[list[float]]:
    batch_dict = self.tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(self.device)
    outputs = self.model(**batch_dict)
    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()
    