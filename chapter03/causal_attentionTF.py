from keras import Layer, layers
import tensorflow as tf


class CausalAttention(Layer):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.query = layers.Dense(d_out, use_bias=qkv_bias)
        self.key = layers.Dense(d_out, use_bias=qkv_bias)
        self.value = layers.Dense(d_out, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout)
        self.context_length = d_out
        self.masked = tf.cast(
            tf.linalg.band_part(
                tf.ones(shape=(context_length, context_length)),
                -1,
                0
            ),
            dtype=tf.float32
        )

    def call(self, x, *args, **kwargs):
        query = self.query(x)  # --> [batch size, S, d_out ]
        key = self.key(x)  # --> [batch size, S, d_out ]
        val = self.value(x)  # --> [batch size, S, d_out ]

        attn_scores = tf.matmul(query, tf.transpose(key, perm=(0, 2, 1)))

        mask = attn_scores * self.masked + ( 1 - self.masked ) * (-1e9)
        d_k = tf.cast(tf.shape(key)[-1], dtype=tf.float32)
        attn_weights = mask / tf.math.sqrt(d_k)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        context_vec = tf.matmul(attn_weights, val)

        return context_vec
