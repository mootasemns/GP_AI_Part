from sentence_transformers import SentenceTransformer

class SentenceTransformerModel:
    def __init__(self, model_name, device):
        self.device = device
        self.model = SentenceTransformer(model_name).to(self.device)

    def encode(self, sentences):
        return self.model.encode(sentences)
