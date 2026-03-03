from GPTArchitecture import GPTModel, TransformerBlock
import json

if __name__ == '__main__':
    GPT_CONFIG_124M = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))

    model_medium = GPTModel({
      "vocab_size": 50207,
      "context_length": 1024,
      "emb_dim": 1024,
      "n_heads": 16,
      "n_layers": 24,
      "drop_rate": 0.1,
      "qkv_bias": False
    })
    model_large = GPTModel(
        {
            "vocab_size": 50207,
            "context_length": 1024,
            "emb_dim": 1280,
            "n_heads": 20,
            "n_layers": 36,
            "drop_rate": 0.1,
            "qkv_bias": False
        }
    )

    para_medium = sum(p.numel() for p in model_medium.parameters())
    para_large = sum(p.numel() for p in model_large.parameters())

    print(f'Parameters (Medium): {para_medium}\tspace cost: {para_medium * 4 / 1024 ** 3} GB')
    print(f'Parameters (Large): {para_large}\tspace cost: {para_large * 4 / 1024 ** 3} GB')

    '''
    Parameters (Medium): 406110208	space cost: 1.51287841796875 GB
    Parameters (Large): 838092800	space cost: 3.1221389770507812 GB
    '''
