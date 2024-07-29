# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F

class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.n_codes = n_codes
        self.embedding_dim = embedding_dim

    def forward(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings
