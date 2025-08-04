import torch
import torch.nn as nn
import math

class LearnedEncodingModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, nhead: int, nlayers: int, nhid: int, dropout: float = 0.5):
        super(LearnedEncodingModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = nn.Linear(embedding_dim, vocab_size)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [batch_size, seq_len, embedding_dim]
        x = x + self.pe[None, :x.size(1), :]
        return self.dropout(x)

class Autoencoder(nn.Module):
    def __init__(self, original_dim: int, compressed_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(original_dim, compressed_dim))
        self.decoder = nn.Sequential(nn.Linear(compressed_dim, original_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoencoderBaselineModel(nn.Module):
    def __init__(self, vocab_size: int, original_embedding_dim: int, compressed_embedding_dim: int,
                 nhead: int, nlayers: int, nhid: int, autoencoder: Autoencoder, dropout: float = 0.5):
        super(AutoencoderBaselineModel, self).__init__()
        self.full_embedding = nn.Embedding(vocab_size, original_embedding_dim)
        self.autoencoder_encoder = autoencoder.encoder
        for param in self.autoencoder_encoder.parameters():
            param.requires_grad = False
        self.pos_encoder = PositionalEncoding(compressed_embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(compressed_embedding_dim, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding_dim = compressed_embedding_dim
        self.decoder = nn.Linear(compressed_embedding_dim, vocab_size)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.full_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        src_full_emb = self.full_embedding(src)
        with torch.no_grad():
            src_compressed = self.autoencoder_encoder(src_full_emb)
        src = src_compressed * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
