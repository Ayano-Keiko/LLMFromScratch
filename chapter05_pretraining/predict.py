import tensorflow as tf
from text_id_convertion import text_to_id, id_to_text
from generate_text_simple import generate
import tiktoken
import json
from chapter04_LLM_arch.GPTArchitecture import GPTModel

if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding('gpt2')
    config = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))
    dummy_inputs = tf.zeros(shape=(1, config.get('context_length')))

    model = GPTModel(config)
    model(dummy_inputs)

    model.load_weights('../save_weights/pretrain.keras')

    text = "Every effort moves you"
    ids = text_to_id(text, tokenizer, model)
    print(f'Original text: {text}')
    generated_text = generate(model, text,
                            tokenizer=tokenizer,
                             max_new_tokens=6,
                             context_size=20,
                             temperature=5,
                             top_k=3)

    print(f'Generated text: {generated_text}')