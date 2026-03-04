import torch
import tensorflow as tf
from multihead_attention import MultiHeadAttention

if __name__ == '__main__':
    torch.manual_seed(123)

    inputs = torch.tensor([
      [0.43, 0.15, 0.89],  # Your x^1
      [0.55, 0.87, 0.66],  # journey x^2
      [0.57, 0.85, 0.64],  # starts x^3
      [0.22, 0.58, 0.33],  # with x^4
      [0.77, 0.25, 0.10],  # one x^5
      [0.05, 0.80, 0.55]   # step x^6
    ])
    batch = tf.stack([inputs, inputs], axis=0)

    mha = MultiHeadAttention(inputs.shape[1],
                             3,
                             context_length=batch.shape[1],
                             dropout=0.5,
                             num_heads=3,
                             qkv_bias=False)
    out = mha(batch)
    print(out)
    exit()

    # 1) Compute attention scores
    attn_score = torch.empty((inputs.shape[0], inputs.shape[0]))
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_score[i, j] = torch.dot(x_i, x_j)

    # 2) Compute attention weights
    attn_weights = torch.softmax(attn_score, dim=-1)

    # 3) Compute context vectors
    all_context_vector = attn_weights @ inputs

    # print(all_context_vector)

    # sa_v1 = self_attention.SelfAttention_v1(inputs.shape[1], 2)
    # print(sa_v1(inputs))

    # sa_v2 = self_attention.SelfAttention_v2(inputs.shape[1], 2)
    # print(sa_v2(inputs))

    # tensorflow
    # tf_model = self_attention.SelfAttention_tf(inputs.shape[1], 2)
    # print(tf_model(inputs))

    batch = torch.stack((inputs, inputs), dim=0)

    # import causal_attention
    # print(batch.shape)
    # context_len = batch.shape[1]
    # ca = causal_attention.CausalAttention(inputs.shape[1], 2, context_len, 0.0)
    # context_vec = ca(batch)
    #
    # print(context_vec.shape)

    from multihead_attention import MultiHeadAttention
    batch_size, context_size, d_in = batch.shape
    d_out = 2

    mha = MultiHeadAttention(d_in, d_out, context_size, 0.0, num_heads=2)
    context_vec = mha(batch)

    print(context_vec)
    print(context_vec.shape)


