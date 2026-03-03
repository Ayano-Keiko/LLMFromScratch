import torch
import json
import tiktoken
import loss
from chapter02.dataset import create_dataloader_v1
from chapter04_LLM_arch.GPTArchitecture import GPTModel

if __name__ == '__main__':
    file_path = "../the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    config = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))
    tokenizer = tiktoken.get_encoding('gpt2')

    # total_characters = len(text_data)
    # total_tokens = len(tokenizer.encode(text_data))
    # print("Characters:", total_characters)
    # print("Tokens:", total_tokens)

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=True,
        shuffle=True,
    )
    num_workers = 0
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    model = GPTModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        train_loss = loss.calc_loss_loader(train_loader, model, device)
        val_loss = loss.calc_loss_loader(val_loader, model, device)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)