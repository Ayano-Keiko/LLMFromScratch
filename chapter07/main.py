import time

from utils import format_instruction, PromptType
import keras
import tensorflow as tf
from finetune_dataset import InstructionDataset
import json
import tiktoken
from configs_finetune import BASE_CONFIG, model_configs, CHOOSE_MODEL
from chapter06.gpt_download import download_and_load_gpt2
from chapter06.load_weights import load_weights_into_gpt
from chapter04_LLM_arch.GPTArchitecture import GPTModel
from utils import format_input
from chapter05_pretraining import text_id_convertion
from chapter05_pretraining.generate_text_simple import generate


if __name__ == '__main__':
    file_path = "../data/instruction-data.json"

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    with open('../fine_tuning.json', mode='r', encoding='UTF-8') as fp:
        configs = json.load(fp)

    # print('Configs')
    # print(configs)
    # print('\n\n')
    BASE_CONFIG.update(configs)
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    print('Model configs')
    print(BASE_CONFIG)
    print('\n\n')

    context_length = configs.get('context_length')
    batch_size = configs.get('batch_size')
    ignore_index = configs.get('ignore_index')
    epochs = configs.get('epochs', 1)
    epochs = 1

    # print(format_instruction(data[50], prompt_type=PromptType.Phi_3))

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    tokenizer = tiktoken.get_encoding("gpt2")
    pad_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})

    train_dataset = InstructionDataset(train_data, tokenizer, batch_size=batch_size, pad_token_id=pad_id, allowed_max_length=context_length, ignore_index=ignore_index)
    test_dataset = InstructionDataset(test_data, tokenizer, batch_size=batch_size, pad_token_id=pad_id, allowed_max_length=context_length, ignore_index=ignore_index)
    val_dataset = InstructionDataset(val_data, tokenizer, batch_size=batch_size, pad_token_id=pad_id, allowed_max_length=context_length, ignore_index=ignore_index)

    batch0 = train_dataset.__getitem__(0)

    print(f'Training batch: {train_dataset.__len__()}')
    print(f'Validation batch: {val_dataset.__len__()}')
    print(f'Test batch: {test_dataset.__len__()}')

    # print(f'source: {batch0[0].shape}')  (16, 74)
    # print(f'target: {batch0[1].shape}')  (16, 74)

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="../save_weights/gpt2"
    )

    fine_tune = GPTModel(BASE_CONFIG)
    # build the model
    dummy_inputs = tf.zeros(shape=(2, BASE_CONFIG['context_length']))
    fine_tune(dummy_inputs)

    load_weights_into_gpt(fine_tune, params, pretrain=True)

    # model architecture
    fine_tune.summary()

    # Before we start finetuning the model in the next section, let's see how it performs on one of the validation tasks
    # print('\n\n==== Before Fine Tuning ====\n')
    # input_text = format_input(val_data[0].get('instruction'), val_data[0].get('input'))
    # print(input_text)
    #
    # generated_text = generate(
    #     model=fine_tune,
    #     text=input_text,
    #     tokenizer=tokenizer,
    #     max_new_tokens=35,
    #     context_size=BASE_CONFIG["context_length"],
    #     eos_id=50256,
    # )
    # response_text = (
    #     generated_text[len(input_text):]
    #     .replace("### Response:", "")
    #     .strip()
    # )
    # print('Predicted results')
    # print(response_text)

    print('\n\n==== Training Process ====')

    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        ignore_class=ignore_index
    )
    optimizer = keras.optimizers.AdamW(learning_rate=1e-4)

    fine_tune.compile(
        optimizer = optimizer,
        loss = loss_fn,
        metrics=[]
    )
    start_time = time.time()
    fine_tune.fit(
        train_dataset,
        validation_data=val_dataset
    )
    finish_time = time.time()

    print(f'Training time: {(finish_time - start_time) / 60} mins')

    fine_tune.save('../save_weights/Fine-Tuning.keras')