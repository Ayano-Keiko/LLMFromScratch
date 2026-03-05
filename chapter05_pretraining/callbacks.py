import tensorflow as tf
import tiktoken
from text_id_convertion import text_to_id

class LLMCallBack(tf.keras.callbacks.Callback):
    def __init__(self, start_inputs, max_new_tokens, context_len):
        super().__init__()
        self.start_inputs = start_inputs
        self.max_new_tokens = max_new_tokens
        self.context_len = context_len

    def on_epoch_end(self, epoch, logs=None):

        tokenizer = tiktoken.get_encoding('gpt2')
        idx = text_to_id(self.start_inputs, tokenizer, self.model)

        for i in range(self.max_new_tokens):
            idx_cond = idx[:, -self.context_len:]
            out = self.model(idx_cond)
            logits = out[:, -1, :]
            probas = tf.nn.softmax(logits, dim=-1)
            idx_next = tf.math.argmax(probas, dim=-1, keepdim=True)
            idx = tf.concat([idx, idx_next], axis=1)

        text_res = tokenizer.decode(idx[0])
        print(f'Epoch: {epoch} generate text: {text_res}')