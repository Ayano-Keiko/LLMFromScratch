import numpy
import tensorflow as tf
from chapter05_pretraining.text_id_convertion import text_to_id, id_to_text

def generate(model, text, tokenizer, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None) -> str:
    idx = text_to_id(text, tokenizer, model)

    for _ in range(max_new_tokens):
        # generate text in loop
        idx_cond = idx[:, -context_size:]
        logits = model.predict(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k:
            topk = tf.math.top_k(logits)
            min_val = topk.values[:, -1]
            logits = tf.where(logits < min_val, tf.fill(tf.shape(logits), -numpy.inf), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits /= temperature

            # Apply softmax to get probabilities
            probs = tf.nn.softmax(logits, axis=-1)
            idx_next = tf.random.categorical(probs, num_samples=1, dtype=idx.dtype)
        else:
            idx_next = tf.argmax(logits, axis=-1, output_type=idx.dtype)
            idx_next = tf.expand_dims(idx_next, axis=-1)

        # Stop generating early if end-of-sequence token is encountered and eos_id is specified
        if eos_id is not None and tf.reduce_any(idx_next == eos_id):
            break

        idx = tf.concat([idx, idx_next], axis=1)


    return id_to_text(idx, tokenizer, model)
