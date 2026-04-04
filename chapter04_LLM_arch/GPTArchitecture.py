import math
import tensorflow
import tensorflow as tf
import keras
import numpy
from chapter03.multihead_attention import MultiHeadAttention
from matplotlib import pyplot as plt

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

    def get_config(self):
        cfg = super().get_config()

        return cfg

@keras.saving.register_keras_serializable()
class GPTModel(keras.Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = config
        self.tok_emb = keras.layers.Embedding(input_dim=config['vocab_size'], output_dim=config['emb_dim'])
        self.pos_emb = keras.layers.Embedding(input_dim=config['context_length'], output_dim=config['emb_dim'])
        self.drop_emb = keras.layers.Dropout(config['drop_embed'])
        self.transformers = [
            TransformerBlock(config) for _ in range(config['n_layers'])
        ]
        self.final_norm = LayerNorm(config['emb_dim'])
        self.out_head = keras.layers.Dense(config['vocab_size'])

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

        x = self.final_norm(x)
        logits = self.out_head(x)

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


if __name__ == '__main__':
    x = numpy.linspace(-5, 5, 100)

    res = GELU().call(x)


    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, res)
    ax.grid()

    plt.show()