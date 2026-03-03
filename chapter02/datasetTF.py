import tensorflow as tf

class GPTDataset:
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

