import torch
import torch.nn as nn
import torch.nn.functional as F

class MRQSelfEvaluator:
    def __init__(self, text_encoder, value_predictor, embedding_model):
        self.encoder = text_encoder
        self.value_predictor = value_predictor
        self.embedding_model = embedding_model  # e.g., Sentence-BERT

    def evaluate(self, prompt, output_a, output_b):
        prompt_emb = self.embedding_model.encode(prompt)
        output_a_emb = self.embedding_model.encode(output_a)
        output_b_emb = self.embedding_model.encode(output_b)

        zs = self.encoder.zs(torch.tensor(prompt_emb).unsqueeze(0))
        zsa_a = self.encoder(zs, torch.tensor(output_a_emb).unsqueeze(0))
        zsa_b = self.encoder(zs, torch.tensor(output_b_emb).unsqueeze(0))

        value_a = self.value_predictor(zsa_a).item()
        value_b = self.value_predictor(zsa_b).item()

        preferred = output_a if value_a >= value_b else output_b
        return preferred, {"value_a": value_a, "value_b": value_b}
