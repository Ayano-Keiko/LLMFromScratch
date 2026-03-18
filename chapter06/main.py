import json

import keras
import tensorflow as tf
import numpy
import tiktoken
from chapter02.dataset import SpamDataset
from chapter04_LLM_arch.GPTArchitecture import GPTModel
from load_weights import load_weights_into_gpt
from gpt_download import download_and_load_gpt2
from chapter05_pretraining.text_id_convertion import text_to_id, id_to_text
from chapter05_pretraining.generate_text_simple import generate

class SpamTextModel(keras.Model):
    def __init__(self, num_class, config):
        super().__init__()
        self.baseModel = GPTModel(config)
        self.classifer = keras.layers.Dense(num_class, activation=keras.activations.softmax)

    def call(self, x, *args, **kwargs):
        x = self.baseModel(x)
        return self.classifer(x)

if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding('gpt2')
    # print(tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'}))
    config = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))
    config.update(config['model_configs'][config.get('CHOOSE_MODEL')])

    config['batch_size'] = 8
    config['epochs'] = 5

    train_dataset = SpamDataset(
        csv_file="../data/sms_spam_collection/train.csv",
        tokenizer=tokenizer,
        batch_size=config.get('batch_size'),
        max_length=None
    )

    valid_dataset = SpamDataset(
        csv_file='../data/sms_spam_collection/validation.csv',
        tokenizer=tokenizer,
        batch_size=config.get('batch_size'),
        max_length=None
    )

    print(train_dataset.max_length)

    dummp_inputs = tf.zeros(shape=(1, config.get('context_length')), dtype=tf.int32)
    model_size = config.get('CHOOSE_MODEL').split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="../save_weights/gpt2")

    gpt_model = GPTModel(config)
    gpt_model(dummp_inputs)  # build the model
    load_weights_into_gpt(gpt_model, params)

    gpt_model.trainable = False  # freeze all GPT weights

    inputs = keras.Input(shape=(None,), dtype=tf.int32)
    gpt_outputs = gpt_model(inputs)  # shape: (batch_size, seq_len, vocab_size)

    # Option 1: Use only the last token's representation
    last_token_logits = gpt_outputs[:, -1, :]  # (batch_size, vocab_size)
    x = keras.layers.Dense(2, activation='softmax')(last_token_logits)  # classification head

    model = keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=4e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    inputs = tokenizer.encode("Do you have time")
    inputs = tf.expand_dims(inputs, axis=0)

    outputs = model.predict(inputs)
    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape)  # shape: (batch_size, num_tokens, num_classes)

    model.fit(train_dataset,
              validation_data=valid_dataset,
              epochs=config['epochs'])

    '''
    
    '''

