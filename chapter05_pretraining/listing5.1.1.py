import json
import tiktoken

import text_id_convertion
from chapter04_LLM_arch.generate_text_simple import generate_text_simple
from chapter04_LLM_arch.GPTArchitecture import GPTModel

if __name__ == '__main__':
    config = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))
    start_context = 'Every effort moves you'
    tokenizer = tiktoken.get_encoding('gpt2')

    model = GPTModel(config)

    tokenIDs = text_id_convertion.text_to_id(start_context, tokenizer, model)
    print(f'Input shape: {tokenIDs.shape}')
    model.eval()
    out_ids = generate_text_simple(model, tokenIDs, 10, context_size=config['context_length'])
    out_text = text_id_convertion.id_to_text(out_ids, tokenizer, model)

    print(f'Output shape: {out_ids.shape}')
    print(f'Generated text: {out_text}')

