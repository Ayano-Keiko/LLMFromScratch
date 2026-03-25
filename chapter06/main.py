import json

import keras
import tensorflow as tf
import numpy
import tiktoken
from chapter02.dataset import SpamDataset
from classification_model import SpamClassify
from load_weights import load_weights_into_gpt
from gpt_download import download_and_load_gpt2
from chapter05_pretraining.text_id_convertion import text_to_id, id_to_text
from chapter05_pretraining.generate_text_simple import generate


if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding('gpt2')
    # print(tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'}))
    config = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))
    config.update(config['model_configs'][config.get('CHOOSE_MODEL')])

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

    spam_classify = SpamClassify(config)
    spam_classify(dummp_inputs)  # build the model
    load_weights_into_gpt(spam_classify, params, pretrain=False)  # no out_head layer

    # spam_classify.trainable = False  # freeze all GPT weights
    # spam_classify.classify.trainable = True

    num_layers = len(spam_classify.layers)
    for layer in range(num_layers):
        if num_layers < len(spam_classify.layers) - 1:
            layer.trainable = False

    inputs = keras.Input(shape=(None,), dtype=tf.int32)
    gpt_outputs = spam_classify(inputs)  # shape: (batch_size, seq_len, vocab_size)


    spam_classify.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=4e-4),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    inputs = tokenizer.encode("Do you have time")
    inputs = tf.expand_dims(inputs, axis=0)

    outputs = spam_classify.predict(inputs)
    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape)  # shape: (batch_size, num_tokens, num_classes)

    spam_classify.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=config['epochs'] # config['epochs']
    )

