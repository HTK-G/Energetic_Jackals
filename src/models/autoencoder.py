"""PyTorch autoencoder used for learning song embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.constants import DEMO_RANDOM_SEED


@dataclass(slots=True)
class AutoencoderConfig:
    """Configuration for the feed-forward autoencoder."""

    input_dim: int
    embedding_dim: int = 12
    hidden_dims: tuple[int, int] = (128, 64)
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 20
    random_state: int = DEMO_RANDOM_SEED


class _FeedForwardAutoencoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dims: tuple[int, int]) -> None:
        super().__init__()

        encoder_layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            previous_dim = hidden_dim
        encoder_layers.append(nn.Linear(previous_dim, embedding_dim))

        decoder_layers: list[nn.Module] = []
        previous_dim = embedding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            previous_dim = hidden_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(inputs)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


class AutoencoderEmbeddingModel:
    """Train and serve low-dimensional embeddings from a dense autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        self.config = config
        self.device = torch.device("cpu")
        self.model_: _FeedForwardAutoencoder | None = None
        self.training_history_: list[float] = []

    def fit(self, feature_matrix: np.ndarray) -> "AutoencoderEmbeddingModel":
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
        torch.manual_seed(self.config.random_state)

        dataset = TensorDataset(torch.from_numpy(feature_matrix))
        batch_size = min(self.config.batch_size, max(len(dataset), 1))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model_ = _FeedForwardAutoencoder(
            input_dim=self.config.input_dim,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.config.learning_rate)
        loss_function = nn.MSELoss()
        self.training_history_ = []

        for _ in range(self.config.epochs):
            epoch_loss = 0.0
            self.model_.train()

            for (batch_inputs,) in dataloader:
                batch_inputs = batch_inputs.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                _, reconstruction = self.model_(batch_inputs)
                loss = loss_function(reconstruction, batch_inputs)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_inputs)

            self.training_history_.append(epoch_loss / len(dataset))

        return self

    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("AutoencoderEmbeddingModel must be fitted before calling transform().")

        self.model_.eval()
        inputs = torch.as_tensor(feature_matrix, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            embeddings = self.model_.encoder(inputs).cpu().numpy()
        return embeddings

    def fit_transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        self.fit(feature_matrix)
        return self.transform(feature_matrix)
