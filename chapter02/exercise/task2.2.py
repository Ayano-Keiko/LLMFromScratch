'''
## Exercise 2.2 Data loaders with different strides and context sizes

To develop more intuition for how the data loader works, try to run it with different
settings such as max_length=2 and stride=2, and max_length=8 and stride=2.
'''

import tiktoken
from chapter02.dataset import GPTDataset_v2

if __name__ == "__main__':
    print("max_length=2 and stride=2")

    dataset1 = GPTDataset_v2(
        '../data/the-verdict.txt',
        encoding='utf-8',
        max_length=2,
        stride=2
    )

    src, tgt = dataset1.__getitem__(0)
    print(f'{src} --> {tgt}')

    print(f'Source: {tokenizer.decode(src)}')
    print(f'Target: {tokenizer.decode(tgt)}')

    print("max_length=8 and stride=2")

    dataset2 = GPTDataset_v2(
        '../data/the-verdict.txt',
        encoding='utf-8',
        max_length=8,
        stride=2
    )

    src, tgt = dataset2.__getitem__(0)
    print(f'{src} --> {tgt}')

    print(f'Source: {tokenizer.decode(src)}')
    print(f'Target: {tokenizer.decode(tgt)}')