import numpy as np
import torch
import logging
from torch import nn
from torch.utils.data import DataLoader
from lightcone.app import run


logger = logging.getLogger(__name__)


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

    def _get_device(self):
        return next(self.parameters()).device

    def explore(self, data_loader: DataLoader, device=None):
        """
        """

        if device is None:
            logger.warning('No device given, try to read off from parameters')
            device = self._get_device()

        run(
            model=self,
            embedding=self.get_embedding(data_loader, device=device),
            device=device,
        )
