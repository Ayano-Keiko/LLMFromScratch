import torch


class CausalAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask',
                             torch.triu(torch.ones(context_length, context_length), diagonal=1)
                             )

    def forward(self, x):
        b, num_token, d_in = x.shape

        key = self.W_key(x)
        query = self.W_query(x)
        value = self.W_value(x)

        attn_score = query @ key.transpose(1, 2)
        attn_score.masked_fill_(
            self.mask.bool()[:num_token, :num_token], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_score / key.shape[-1] ** 0.5,
            dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ value

        return context_vec