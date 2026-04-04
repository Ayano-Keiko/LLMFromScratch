import numpy
import tensorflow as tf
import keras
from utils import format_instruction, PromptType

class InstructionDataset(keras.utils.PyDataset):
    def __init__(self,  data, tokenizer, batch_size, allowed_max_length = None, ignore_index=-100, pad_token_id=50256):
        super().__init__()
        self.data = data
        self.encoded_text = []
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.allowed_max_length = allowed_max_length
        self.ignore_index = ignore_index

        for entry in data:
            # Step1: Format data using prompt template.
            full_text = format_instruction(entry, prompt_type=PromptType.Alpaca)
            # Step2: Tokenize formatted data.
            idxs = tokenizer.encode(full_text)
            if allowed_max_length is not None:
                idxs = idxs[:allowed_max_length]
            self.encoded_text.append(idxs)

    def __len__(self):
        return len(self.encoded_text) // self.batch_size

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.encoded_text))
        # Step 3: Adjust to the same length with padding tokens.
        batch = self.encoded_text[low: high]
        pad_batch = tf.keras.utils.pad_sequences(batch, padding='post', value=self.pad_token_id)  # ndarray (batch, max seq length) (2, 74)
        # print(pad_batch.shape); print(pad_batch[:, 1:].shape)  # (2, 74) (2, 73)
        # Step 4: Create target token IDs for training
        # Similar to pretraining an LLM, the targets are the inputs shifted by 1 position to the right, so the LLM learns to predict the next token
        trg_batch = tf.keras.utils.pad_sequences(pad_batch[:, 1:], padding='post', value=self.pad_token_id, maxlen=pad_batch.shape[1]) # (2, 74)
        # print(trg_batch.shape)  # (2, 74)
        # Step 5: Replace padding tokens with placeholders.
        # get mask
        mask = (trg_batch == self.pad_token_id).astype(int)
        # turn first occurred 1 to 0
        first_indices = numpy.argmax(mask == 1, axis=1)
        has_one = numpy.any(mask == 1, axis=1)
        rows_to_update = numpy.where(has_one)[0]
        mask[rows_to_update, first_indices[rows_to_update]] = 0
        trg_batch[mask == 1] = self.ignore_index

        # print(pad_batch[1]); print(trg_batch[1])

        return tf.convert_to_tensor(pad_batch, dtype=tf.int32), tf.convert_to_tensor(trg_batch, dtype=tf.int32)
