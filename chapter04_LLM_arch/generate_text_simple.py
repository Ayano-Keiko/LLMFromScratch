import tensorflow as tf
import torch
import numpy

def generate_text_simple(model, idx,
    max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate_text_simpleTF(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size: ]
        logits = model.call(idx_cond)

        logits = logits[:, -1, :]  # (1, 50207)
        prob = tf.nn.softmax(logits, axis=-1)  # (1, 50207)
        index_next = numpy.argmax(prob, axis=-1, keepdims=True)
        idx = tf.concat([idx, index_next], axis=1)

    return idx