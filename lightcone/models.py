import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightcone.app import run


class AutoEncoder(nn.Module):
    """
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        embedding = self.encoder(x)
        return self.decoder(embedding)

    def get_embedding(self, data_loader: DataLoader, device='cpu'):
        """
        """
        with torch.no_grad():
            return np.concatenate(
                [
                    self.encoder(
                        X.to(device)
                    ).detach().cpu().numpy() for X in data_loader
                ]
            )

    def explore(self, data_loader: DataLoader = None, device='cpu'):
        """
        """
        if data_loader is not None:
            embedding = self.get_embedding(data_loader, device='cpu')
        else:
            embedding = None
        run(model=self, embedding=embedding)
