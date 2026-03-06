import keras
import tensorflow as tf

@keras.saving.register_keras_serializable()
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length

        self.query = keras.layers.Dense(d_out, use_bias=qkv_bias)
        self.key = keras.layers.Dense(d_out, use_bias=qkv_bias)
        self.value = keras.layers.Dense(d_out, use_bias=qkv_bias)
        self.dropout = keras.layers.Dropout(dropout)
        self.masked = self.add_weight(
            name="causal_mask",
            shape=(context_length, context_length),
            initializer=tf.keras.initializers.Constant(
                tf.linalg.band_part(tf.ones((context_length, context_length)), -1, 0)
            ),
            trainable=False
        )

    def call(self, x, *args, **kwargs):
        shape = tf.shape(x)  # (2, 6, 3)
        b, num_tokens, d_in = shape[0], shape[1], shape[2]

        query = self.query(x)
        key = self.key(x)
        val = self.value(x)

        # (b, num_tokens, self.num_heads, self.head_dim)
        query = tf.reshape(query, shape=(b, num_tokens, self.num_heads, self.head_dim))
        key = tf.reshape(key, shape=(b, num_tokens, self.num_heads, self.head_dim))
        val = tf.reshape(val, shape=(b, num_tokens, self.num_heads, self.head_dim))

        # (b, self.num_heads, num_tokens, self.head_dim)
        query = tf.transpose(query, perm=(0, 2, 1, 3))
        key = tf.transpose(key, perm=(0, 2, 1, 3))
        val = tf.transpose(val, perm=(0, 2, 1, 3))

        attn_scores = tf.matmul(query, tf.transpose(key, perm=(0, 1, 3, 2)))

        # mask = attn_scores * self.masked + (1 - self.masked) * -1e9
        mask_cut = self.masked[:num_tokens, :num_tokens]
        mask_cut = tf.cast(mask_cut, dtype=tf.bool)
        attn_scores = tf.where(mask_cut, attn_scores, -1e9)

        d_k = tf.cast(self.head_dim, dtype=tf.float32)
        attn_weights = tf.nn.softmax(attn_scores / tf.math.sqrt(d_k), axis=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = tf.matmul(attn_weights, val)  # (2, 3, 6, 1)
        context_vec = tf.transpose(context_vec, perm=(0, 2, 1, 3))
        context_vec = tf.reshape(context_vec, shape=(b, num_tokens, self.num_heads * self.head_dim))

        return context_vec

    def get_config(self):
        cfg = super().get_config()

        return cfg
