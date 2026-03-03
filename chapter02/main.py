import dataset
from dataset import create_dataloader_v1
import datasetTF
from datasetTF import GPTDataset
import tiktoken

if __name__ == '__main__':
    # with open('../the-verdict.txt', 'r', encoding='utf-8') as f:
    #     raw_text = f.read()
    #
    # dataloader = create_dataloader_v1(
    #     raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    # )
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset('../the-verdict.txt', 'utf-8', tokenizer=tokenizer, max_length=4, stride=1)
    source, target = dataset.getDataset()

    for src, trg in zip(source, target):
        print(src, trg)

    # iterator = iter(dataloader)
    # first_batch = next(iterator)
    # print(first_batch)
