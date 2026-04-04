import tensorflow as tf
from keras import Layer, layers


class SelfAttention(Layer):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.W_q = layers.Dense(d_out, use_bias=qkv_bias)  # self.add_weight(shape=(d_in, d_out))
        self.W_k = layers.Dense(d_out, use_bias=qkv_bias)  # self.add_weight(shape=(d_in, d_out))
        self.W_v = layers.Dense(d_out, use_bias=qkv_bias)  # self.add_weight(shape=(d_in, d_out))


    def call(self, x, *args, **kwargs):
        keys = self.W_k(x)  # [ L, d_out ]
        query = self.W_q(x)  # [ L, d_out ]
        value = self.W_v(x)  # [ L, d_out ]

        attn_scores = tf.linalg.matmul(keys, tf.transpose(query, perm=[1, 0]))

        attn_weights = tf.nn.softmax(attn_scores / tf.math.sqrt(tf.constant([keys.shape[-1]], dtype=tf.float32)),
                                     axis=-1)
        context_vec =  tf.linalg.matmul(attn_weights, value)
        return context_vec
