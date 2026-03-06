import tensorflow as tf
from text_id_convertion import text_to_id, id_to_text
from chapter04_LLM_arch.generate_text_simple import generate_text_simple
import tiktoken
import json

if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding('gpt2')
    config = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))

    model = tf.keras.models.load_model('../save_weights/pretrain.keras')
    text = "Every effort moves you"
    ids = text_to_id(text, tokenizer, model)
    generated_ids = generate_text_simple(model, ids, 6, 20)
    print('Generated text: ', id_to_text(generated_ids, tokenizer, model))