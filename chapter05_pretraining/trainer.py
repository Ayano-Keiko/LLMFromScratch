import json
import tiktoken
import keras
from chapter02.dataset import GPTDataset_v1, GPTDataset_v2
from chapter04_LLM_arch.GPTArchitecture import GPTModel
from callbacks import LLMCallBack
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # print(keras.backend.backend())
    # print(tf.test.gpu_device_name())
    # print(tf.config.list_physical_devices('gpu'))

    tokenizer = tiktoken.get_encoding('gpt2')
    config = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))
    epochs = config.get('epochs')

    dataset = GPTDataset_v2('../the-verdict.txt',
                            encoding='utf-8',
                            max_length=config.get('context_length'),
                            stride=config.get('context_length'),
                            batch_size=config.get('batch_size')
                         )
    # print(dataset.__getitem__(0)[0].shape)
    callback = LLMCallBack('Every effort moves you', max_new_tokens=6, context_len=config.get('context_length'))

    model = GPTModel(config)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=4e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    history = model.fit(dataset,
                        epochs=epochs,
                        callbacks=[callback])

    model.save('../save_weights/pretrain.keras')

    # model.load_weights('../save_weights/pretrain.keras')

    epochs_range = [i + 1 for i in range(epochs)]

    fig = plt.figure(figsize=(25, 16))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_title("accuracy")
    ax.plot(epochs_range, history.history['accuracy'])
    ax_loss = fig.add_subplot(2, 1, 2)
    ax_loss.set_title('loss')
    ax_loss.plot(epochs_range,  history.history['loss'])
    fig.savefig('./Reports/pretain.svg')