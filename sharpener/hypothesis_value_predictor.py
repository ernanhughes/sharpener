from torch import nn
import torch    

class HypothesisValuePredictor(nn.Module):
    """Predicts a quality score for a hypothesis given its embedding."""
    def __init__(self, zsa_dim, hdim):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(zsa_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, 1)
        )

    def forward(self, zsa_embedding):
        return self.value_net(zsa_embedding)
