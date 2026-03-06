import json
import tensorflow as tf
import tiktoken
import keras
from chapter02.dataset import GPTDataset_v1, GPTDataset_v2
from chapter04_LLM_arch.GPTArchitecture import GPTModel
from chapter04_LLM_arch.generate_text_simple import generate_text_simple
from text_id_convertion import text_to_id, id_to_text

if __name__ == '__main__':
    # print(keras.backend.backend())
    # print(tf.test.gpu_device_name())
    # print(tf.config.list_physical_devices('gpu'))

    tokenizer = tiktoken.get_encoding('gpt2')
    config = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))

    dataset = GPTDataset_v2('../words.txt',
                            encoding='utf-8',
                            max_length=config.get('context_length'),
                            stride=config.get('context_length'),
                            batch_size=config.get('batch_size')
                         )
    # print(dataset.__getitem__(0)[0].shape)


    model = GPTModel(config)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-6),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = model.fit(dataset,
                        epochs=10)

    model.save('../save_weights/pretrainCN.keras')

    # model.load_weights('../save_weights/pretrain.keras')
