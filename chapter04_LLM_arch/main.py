import tensorflow as tf
from GPTArchitecture import GPTModel
from chapter02.dataset import GPTDataset
from generate_text_simple import generate_text_simple
import tiktoken
import json
import numpy


if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding('gpt2')

    GPT_CONFIG_124M = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))
    dataset = GPTDataset('../data/the-verdict.txt',
                         'utf-8', tokenizer, 10, 1)

    txt = "Every effort moves you"
    encoded = tokenizer.encode(txt)
    inputs = tf.expand_dims(tf.constant(encoded, dtype=tf.int32), axis=0)

    print(f'encoded: {inputs.shape}')
    model = GPTModel(config=GPT_CONFIG_124M)

    out = generate_text_simple(model, inputs, 10, GPT_CONFIG_124M['context_length'])
    out_idxs = tf.squeeze(out, axis=0)

    print(f'output: {out.shape}')
    print(f'decoded: {out_idxs.shape}')

    decode_text = tokenizer.decode(out_idxs.numpy().tolist())
    print(f'Decoded text: {decode_text}')
