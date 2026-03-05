from dataset import GPTDataset_v1, GPTDataset_v2
import tiktoken

if __name__ == '__main__':

    tokenizer = tiktoken.get_encoding('gpt2')
    # dataset = GPTDataset_v1('../the-verdict.txt', 'utf-8', tokenizer=tokenizer, max_length=4, stride=1)
    # source, target = dataset.getDataset()

    dataset = GPTDataset_v2(txt_path='../the-verdict.txt', encoding='utf-8', max_length=4, stride=1)

    for item in dataset:
        src, tgt = item
        print(src.shape, tgt.shape); break
