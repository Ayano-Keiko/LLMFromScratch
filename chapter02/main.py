import dataset
import dataset
from dataset import GPTDataset, GPTDataset_v2
import tiktoken

if __name__ == '__main__':

    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset('../the-verdict.txt', 'utf-8', tokenizer=tokenizer, max_length=4, stride=1)
    source, target = dataset.getDataset()

    for src, trg in zip(source, target):
        print(src, trg)