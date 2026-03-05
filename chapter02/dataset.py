import tiktoken
import tensorflow as tf

class GPTDataset_v1:
    def __init__(self, txt_path, read_encoding, tokenizer, max_length, stride, **kwargs):
        super().__init__(**kwargs)
        self.source = []
        self.target = []

        with open(txt_path, mode='r', encoding=read_encoding) as fp:

            content = fp.read()
            token_ids = tokenizer.encode(content)
            for i in range(0, len(token_ids) - max_length, stride):
                self.source.append( token_ids[i: i+max_length] )
                self.target.append( token_ids[i+1: i+max_length+1] )

    def getDataset(self):
        return tf.data.Dataset.from_tensor_slices(self.source), tf.data.Dataset.from_tensor_slices(self.target)

class GPTDataset_v2(tf.keras.utils.PyDataset):
    def __init__(self, txt_path, encoding, max_length, stride):
        super().__init__()
        self.source = []
        self.target = []
        self.tokenizer = tiktoken.get_encoding('gpt2')

        with open(txt_path, mode='r', encoding=encoding) as fp:
            content = fp.read()
            token_ids = self.tokenizer.encode(content)

            for i in range(0, len(token_ids) - max_length, stride):
                self.source.append( tf.constant(token_ids[i: i+max_length]) )
                self.target.append( tf.constant(token_ids[i+1: i+max_length+1]) )

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return (self.source[idx], self.target[idx])