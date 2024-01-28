import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device(0)

class policy_func(nn.Module):
    def __init__(self, state_dim, act_dim, h_dim, context_len, max_timestep=4096, lanten_dim=48, residual=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = act_dim
        self.h_dim = h_dim
        self.context_len = context_len
        self.use_residual = residual

        self.encoder = nn.Sequential(nn.Linear(self.h_dim, 1024),
                                     nn.Dropout(0.1),
                                     nn.ReLU(),
                                     nn.Linear(1024, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256),
                                     nn.Dropout(0.1),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU()
                                     )

        self.mu = nn.Linear(64, lanten_dim)
        self.logvar = nn.Linear(64, lanten_dim)
        self.bn1 = nn.BatchNorm1d(self.context_len)
        self.latent_mapping = nn.Linear(lanten_dim, 64)

        self.decoder = nn.Sequential(nn.Linear(64, 128),
                                     nn.BatchNorm1d(self.context_len),
                                     nn.ReLU(),
                                     nn.Linear(128, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, self.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.h_dim, self.action_dim))
        self.embed_ln = nn.LayerNorm(self.h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, self.h_dim)
        self.embed_state = nn.Linear(state_dim, self.h_dim)
        self.embed_residual_state = nn.Linear(state_dim, 64)
        self.embed_residual_z = nn.Linear(lanten_dim, self.action_dim)
        self.tanh = nn.Tanh()

    def encode(self, x, timesteps):
        B, T, _ = x.shape
        timesteps_embeddings = self.embed_timestep(timesteps.long())
        state_embeddings = self.embed_state(x) + timesteps_embeddings
        residual_x = self.embed_residual_state(x)
        x = state_embeddings.reshape(B, T, self.h_dim)
        encoder = residual_x + self.encoder(x)  # residual connection
        mu, logvar = self.mu(encoder), self.logvar(encoder)
        return mu, logvar

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z, x):
        if self.use_residual:
            residual_z = self.embed_residual_z(z)
            latent_z = self.latent_mapping(z)
            out = self.decoder(latent_z)-residual_z
        else:
            latent_z = self.latent_mapping(z)
            out = self.decoder(latent_z)

        reshaped_out = self.tanh(out).view(x.shape[0], self.context_len, self.action_dim)
        return reshaped_out

    def forward(self, timesteps, states):
        mu, logvar = self.encode(states, timesteps)
        z = self.sample_z(mu, logvar)
        output = self.decode(z, states)
        return output
