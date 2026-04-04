import tiktoken
from chapter02.dataset import GPTDataset_v2

'''
## Exercise 2.1 Byte pair encoding of unknown words

Try the BPE tokenizer from the tiktoken library on the unknown words “Akwirw ier” and print the individual token IDs. Then, call the decode function on each of the resulting integers in this list to reproduce the mapping shown in figure 2.11. Lastly, call the decode method on the token IDs to check whether it can econstruct the original input, “Akwirw ier.”
'''

if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding('gpt2')
    unknown_txt = 'Akwirwier'
    ids = tokenizer.encode(unknown_txt)


    print(f'====  {unknown_txt}  ====')
    for id in ids:
        print(f'{id} --> {tokenizer.decode([id])}')