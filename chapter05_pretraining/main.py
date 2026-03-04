import json

import tensorflow as tf
import tiktoken
import keras
from chapter02.dataset import GPTDataset
from chapter04_LLM_arch.GPTArchitecture import GPTModel
from chapter04_LLM_arch.generate_text_simple import generate_text_simple
from text_id_convertion import text_to_id

if __name__ == '__main__':
    # print(keras.backend.backend())

    tokenizer = tiktoken.get_encoding('gpt2')
    config = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))

    data = GPTDataset('../the-verdict.txt',
                            'utf-8',
                            tokenizer,
                            16,
                            16
                            )
    source, target = data.getDataset()

    dataset = tf.data.Dataset.zip(source, target)
    dataset = dataset.batch(batch_size=16)

    # for batch in dataset.take(5):
    #     print(batch[0].shape, batch[1].shape)

    # data_size = len(dataset)

    # print(data_size)
    train_data, valid_data = tf.keras.utils.split_dataset(dataset, left_size=0.9, shuffle=False)

    model = GPTModel(config)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-6),
        loss=keras.losses.SparseCategoricalCrossentropy(),  # CategoricalCrossentropy
        metrics=['accuracy']
    )
    history = model.fit(train_data, validation_data=valid_data, epochs=20)

    text = "Hello, "
    ids = text_to_id(text, tokenizer, model)
    generate_text_simple(model, ids, 6, 20)