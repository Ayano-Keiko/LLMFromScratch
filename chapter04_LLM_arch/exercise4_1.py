from GPTArchitecture import GPTModel, TransformerBlock
import json

if __name__ == '__main__':
    GPT_CONFIG_124M = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))

    model = GPTModel(GPT_CONFIG_124M)

    transformer = TransformerBlock(GPT_CONFIG_124M)
    total_ff = sum(p.numel() for p in transformer.ff.parameters())
    total_attn = sum(p.numel() for p in transformer.att.parameters())

    print(f'Number of parameters in feed forward: {total_ff}\nNumber of parameters in attention modules: {total_attn}')
    print(f'feed forward space: {total_ff * 4 / (1024 ** 2)} MB\nattention modules space: {total_attn * 4 / 1024 ** 2} MB')
