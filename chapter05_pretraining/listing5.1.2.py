import torch
import json
from chapter04_LLM_arch.GPTArchitecture import GPTModel
import tiktoken

if __name__ == '__main__':
    inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
        [40, 1107, 588]]) # "I really like"]
    targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
        [1107, 588, 11311]]) # " really like chocolate"]

    config = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))
    tokenizer = tiktoken.get_encoding('gpt2')
    model = GPTModel(config)

    model.eval()
    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)

    # print(f"Targets batch 1: {id_to_text(targets[0], tokenizer, model)}")
    # print(f"Outputs batch 1:"
    #       f" {id_to_text(token_ids[0].flatten(), tokenizer, model)}")

    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)
    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)

    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)

    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)