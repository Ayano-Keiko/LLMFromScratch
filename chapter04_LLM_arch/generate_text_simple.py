import tensorflow as tf
import numpy


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size: ]
        logits = model(idx_cond)

        logits = logits[:, -1, :]  # (1, 50207)
        prob = tf.nn.softmax(logits, axis=-1)  # (1, 50207)
        index_next = numpy.argmax(prob, axis=-1, keepdims=True)
        idx = tf.concat([idx, index_next], axis=1)

    return idx