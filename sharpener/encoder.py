import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# TextEncoder for embedding prompts and hypotheses
class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=384, zs_dim=256, za_dim=128, zsa_dim=256, hdim=512):
        super().__init__()
        self.zs_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, zs_dim)
        )
        self.za_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, za_dim)
        )
        self.zsa_mlp = nn.Sequential(
            nn.Linear(zs_dim + za_dim, zsa_dim),
            nn.ReLU(),
            nn.Linear(zsa_dim, zsa_dim)
        )

    def forward(self, prompt_emb, response_emb):
        zs = F.relu(self.zs_mlp(prompt_emb))
        za = F.relu(self.za_mlp(response_emb))
        zsa = self.zsa_mlp(torch.cat([zs, za], dim=1))
        return zsa

# HypothesisValuePredictor for explicit evaluation
class HypothesisValuePredictor(nn.Module):
    def __init__(self, zsa_dim=256, hdim=128):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(zsa_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, 1)
        )

    def forward(self, zsa_embedding):
        return self.value_net(zsa_embedding)

# MRQSelfEvaluator combining encoder and predictor
class MRQSelfEvaluator:
    def __init__(self, device='cpu'):
        self.device = device
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.encoder = TextEncoder().to(self.device)
        self.value_predictor = HypothesisValuePredictor().to(self.device)

    def evaluate(self, prompt, output_a, output_b):
        prompt_emb = torch.tensor(self.embedding_model.encode(prompt), device=self.device).unsqueeze(0)
        output_a_emb = torch.tensor(self.embedding_model.encode(output_a), device=self.device).unsqueeze(0)
        output_b_emb = torch.tensor(self.embedding_model.encode(output_b), device=self.device).unsqueeze(0)

        zsa_a = self.encoder(prompt_emb, output_a_emb)
        zsa_b = self.encoder(prompt_emb, output_b_emb)

        value_a = self.value_predictor(zsa_a).item()
        value_b = self.value_predictor(zsa_b).item()

        preferred_output = output_a if value_a >= value_b else output_b
        scores = {"value_a": value_a, "value_b": value_b}

        return preferred_output, scores
