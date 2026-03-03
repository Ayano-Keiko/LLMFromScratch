import tensorflow as tf
from GPTArchitectureTF import GPTModel
import tiktoken
import json
import numpy

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size: ]
        logits = model.call(idx_cond)

        logits = logits[:, -1, :]  # (1, 50207)
        prob = tf.nn.softmax(logits, axis=-1)  # (1, 50207)
        index_next = numpy.argmax(prob, axis=-1, keepdims=True)
        idx = tf.concat([idx, index_next], axis=1)

    return idx

if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding('gpt2')

    GPT_CONFIG_124M = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))

    txt = "海上升明月，"
    encoded = tokenizer.encode(txt)
    inputs = tf.expand_dims(tf.constant(encoded, dtype=tf.int32), axis=0)

    print(f'encoded: {inputs.shape}')

    model = GPTModel( GPT_CONFIG_124M )

    out = generate_text_simple(model, inputs, 10, GPT_CONFIG_124M['context_length'])
    out_idxs = tf.squeeze(out, axis=0)

    print(f'output: {out.shape}')
    print(f'decoded: {out_idxs.shape}')

    decode_text = tokenizer.decode(out_idxs.numpy().tolist())
    print(f'Decoded text: {decode_text}')
    # Decoded text: 海上升明月， Music 911 wandering wandering wandering wandering wandering wandering wandering wandering