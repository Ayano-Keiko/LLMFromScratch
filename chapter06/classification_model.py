import keras
import tensorflow as tf
import tensorflow
import math

@keras.saving.register_keras_serializable()
class LayerNorm(keras.layers.Layer):
    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 1e-5
        self.scale = self.add_weight(
            shape=(emb_dim, ),
            trainable=True,
            name='scale',
            initializer='ones'
        )
        self.shift = self.add_weight(
            shape=(emb_dim, ),
            trainable=True,
            name='shift',
            initializer='zeros'
        )

    def call(self, x, *args, **kwargs):
        mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        norm_x = (x - mean) / tensorflow.math.sqrt(var + self.eps)

        return tf.math.multiply(self.scale, norm_x) + self.shift

    def get_config(self):
        cfg = super().get_config()

        return cfg

@keras.saving.register_keras_serializable()
class GELU(keras.layers.Layer, ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, x, *args, **kwargs):
        val = tf.math.sqrt(tf.constant(2.0 / math.pi, dtype=tf.float32))
        c = tf.constant(0.044715, dtype=tf.float32)
        power = tf.constant(3.0, dtype=tf.float32)

        return 0.5 * x * (
            1.0 + tf.math.tanh(
                val * (x + c * tf.math.pow(x, power))
            ))

    def get_config(self):
        cfg = super().get_config()

        return cfg

@keras.saving.register_keras_serializable()
class FeedForward(keras.layers.Layer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = config
        self.linear1 = keras.layers.Dense(config['emb_dim'] * 4, activation=keras.activations.gelu)
        self.linear2 = keras.layers.Dense(config['emb_dim'])

    def call(self, x, *args, **kwargs):
        x = self.linear1(x)
        return self.linear2(x)

    def get_config(self):
        cfg = super().get_config()

        return cfg

@keras.saving.register_keras_serializable()
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.out_proj = keras.layers.Dense(d_out)

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

        out = self.out_proj(context_vec)

        return out

    def get_config(self):
        cfg = super().get_config()

        return cfg


@keras.saving.register_keras_serializable()
class TransformerBlock(keras.Layer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.att = MultiHeadAttention(
            d_in=config.get('emb_dim'),
            d_out=config.get('emb_dim'),
            context_length=config.get('context_length'),
            num_heads=config.get('n_heads'),
            dropout=config.get('dropout_mha'),
            qkv_bias=config.get('qkv_bias')
        )
        self.ff = FeedForward(
            config
        )
        self.norm1 = LayerNorm(config.get('emb_dim'))
        self.norm2 = LayerNorm(config.get('emb_dim'))
        self.dropAfmha = keras.layers.Dropout(config.get('dropout_after_mha'))
        self.dropff = keras.layers.Dropout(config.get('dropout_feedforward'))

    def call(self, x, *args, **kwargs):
        short_cut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropAfmha(x)
        x = x + short_cut

        short_cut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropff(x)
        x = x + short_cut

        return x

@keras.saving.register_keras_serializable()
class SpamClassify(keras.Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = config
        self.tok_emb = keras.layers.Embedding(input_dim=config['vocab_size'], output_dim=config['emb_dim'])
        self.pos_emb = keras.layers.Embedding(input_dim=config['context_length'], output_dim=config['emb_dim'])
        self.drop_emb = keras.layers.Dropout(config['drop_embed'])
        self.transformers = [
            TransformerBlock(config) for _ in range(config['n_layers'])
        ]

        self.final_norm = keras.layers.LayerNormalization()
        self.classify = keras.layers.Dense(1, activation=keras.activations.sigmoid)

    def call(self, x, *args, **kwargs):
        # (None, 16)
        # print(x); exit(-1)
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        # tf.print(batch_size, seq_len); exit(-1)

        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(
            tf.range(start=0, limit=seq_len)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        for transLayer in self.transformers:
            x = transLayer(x)

        x = self.final_norm(x)  # (None, 120, 768)
        x = x[:, -1, :]  # last token
        logits = self.classify(x)

        return logits

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "config":  self.cfg
        })
        return cfg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if config.get('config'):
            cfg = config.pop('config')
            return cls(cfg, **config)
        else:
            return cls(**config)