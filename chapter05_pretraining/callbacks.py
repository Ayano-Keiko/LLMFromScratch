import tensorflow as tf
import tiktoken
from text_id_convertion import text_to_id

class LLMCallBackTF(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        start_inputs = 'Every effort moves you'
        max_new_tokens = 6
        context_len = 10

        tokenizer = tiktoken.get_encoding('gpt2')
        idx = text_to_id(start_inputs, tokenizer, self.model)

        for i in range(max_new_tokens):
            idx_cond = idx[:, -context_len:]
            out = self.model(idx_cond)
            logits = out[:, -1, :]
            probas = tf.nn.softmax(logits, dim=-1)
            idx_next = tf.math.argmax(probas, dim=-1, keepdim=True)
            idx = tf.concat([idx, idx_next], axis=1)

        text_res = tokenizer.decode(idx[0])
        print(f'Epoch: {epoch} generate text: {text_res}')