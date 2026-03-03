import torch
from torch import nn
from matplotlib import pyplot as plt
from chapter03.multihead_attention import MultiHeadAttention

GPT_CONFIG_124M = {
    "vocab_size": 50207,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        GELU(),
        nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
                          GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3.0))
        ))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.att = MultiHeadAttention(
            d_in=config['emb_dim'],
            d_out=config['emb_dim'],
            context_length=config['context_length'],
            num_heads=config['n_heads'],
            droupout=config['dropout_mha'],
            qkv_bias=config['qkv_bias']
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config['emb_dim'])
        self.norm2 = LayerNorm(config['emb_dim'])
        self.drop_afmha = nn.Dropout(config['dropout_after_mha'])
        self.drop_ff = nn.Dropout(config['dropout_feedforward'])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_afmha(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_ff(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emb = nn.Embedding(config['context_length'], config['emb_dim'])
        self.drop_emb = nn.Dropout(config['drop_embed'])
        self.transformer_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )
        self.final_norm = LayerNorm(config['emb_dim'])
        self.out_heads = nn.Linear(in_features=config['emb_dim'], out_features=config['vocab_size'], bias=False)

    def forward(self, x):
        batch_size, seq_length = x.shape

        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(
            torch.arange(seq_length, device=x.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer_block(x)
        x = self.final_norm(x)
        logits = self.out_heads(x)

        return logits



class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.get('vocab_size'), cfg.get('emb_dim'))
        self.pos_emb = nn.Embedding(cfg.get('context_length'), cfg.get('emb_dim'))
        self.drop_emb = nn.Dropout(cfg.get('drop_rate'))
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg.get('n_layers'))]
        )
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False
        )

    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=x.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x

if __name__ == '__main__':
    gelu = GELU()
    x = torch.linspace(-5, 5, 100)

    res = gelu(x)


    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, res)
    plt.show()