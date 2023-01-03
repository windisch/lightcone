import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from lightcone.models import AutoEncoder


class ToyDataset(Dataset):

    def __init__(self):
        super(Dataset, self).__init__()

        self.random = np.random.RandomState(10)

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return self.random.normal(size=(1, 20, 20)).astype(np.float32)


class ToyDecoder(nn.Module):

    def __init__(self):
        super(ToyDecoder, self).__init__()

        self.decode = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 20*20)
        )

    def forward(self, x):

        x = self.decode(x)
        x = torch.reshape(x, (x.shape[0], 1, 20, 20))
        return x


class ToyEncoder(nn.Module):

    def __init__(self):
        super(ToyEncoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(20*20, 100),
            nn.Tanh(),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.encode(x)
        return x


class TestAutoencoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.model = AutoEncoder(
            encoder=ToyEncoder(),
            decoder=ToyDecoder(),
        ).to('cpu')

        cls.data = ToyDataset()
        cls.loader = DataLoader(cls.data, batch_size=10)
        optimizer = torch.optim.Adam(cls.model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        for X in cls.loader:
            X = X.to('cpu')
            Y = cls.model(X)
            loss = loss_fn(X, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_reconstruction(self):
        with torch.no_grad():
            X = torch.randn(5, 1, 20, 20)
            Y = self.model(X)
        self.assertEqual(Y.shape, torch.Size([5, 1, 20, 20]))

    def test_embedding(self):
        with torch.no_grad():
            X = torch.randn(5, 1, 20, 20)
            Y = self.model.encoder(X)
        self.assertEqual(Y.shape, torch.Size([5, 2]))

    def test_get_embedding(self):
        embedding = self.model.get_embedding(self.loader)
        self.assertTupleEqual(
            embedding.shape,
            (100, 2)
        )
