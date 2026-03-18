import tiktoken
import tensorflow as tf
import pandas

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
    def __init__(self, txt_path, encoding, max_length, stride, batch_size):
        super().__init__()
        self.source = []
        self.target = []
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.batch_size = batch_size

        with open(txt_path, mode='r', encoding=encoding) as fp:
            content = fp.read()
            token_ids = self.tokenizer.encode(content)

            for i in range(0, len(token_ids) - max_length, stride):
                self.source.append( token_ids[i: i+max_length] )
                self.target.append( token_ids[i+1: i+max_length+1] )

    def __len__(self):
        return len(self.source) // self.batch_size

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.source))

        return tf.constant(self.source[low: high]), tf.constant(self.target[low: high])

class SpamDataset(tf.keras.utils.PyDataset):
    def __init__(self, csv_file, tokenizer, batch_size, max_length=None, pad_token_id=50256):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = pandas.read_csv(csv_file, encoding='UTF-8')
        self.batch_size = batch_size
        self.label_map = {"ham": 0, "spam": 1}

        self.encoded_text = [
           self.tokenizer.encode(text) for text in self.data['text']
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_text = [
                text[:self.max_length] for text in self.encoded_text
            ]

        # Pad sequences to the longest sequence
        self.encoded_text = [
            text + [pad_token_id] * (self.max_length - len(text))
            for text in self.encoded_text
        ]

    def __len__(self):
        return len(self.encoded_text) // self.batch_size

    def __getitem__(self, idx):
        row = idx * self.batch_size
        high = row + self.batch_size

        text = tf.constant(self.encoded_text[row: high])
        label = tf.constant(self.data.iloc[row:high]['labels'], dtype=tf.int32)

        return text, label

    def _longest_encoded_length(self):
        max_leanth = 0

        for text in self.encoded_text:
            max_leanth = max(len(text), max_leanth)

        return max_leanth