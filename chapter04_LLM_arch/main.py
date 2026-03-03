import json

import tiktoken
import torch
import tensorflow as tf
import GPTArchitectureTF
import GPTArchitecture
from chapter04_LLM_arch.generate_text_simple import generate_text_simple

if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding('gpt2')

    batch = []
    txt1 = 'Every effort moves you'
    txt2 = 'Every day holds a'
    tok_txt1 = tokenizer.encode(txt1)
    tok_txt2 = tokenizer.encode(txt2)

    batch.append(torch.tensor(tok_txt1))
    batch.append(torch.tensor(tok_txt2))

    batch = torch.stack(batch, dim=0)

    batchTF = tf.constant([
        tok_txt1, tok_txt2
    ])

    # torch.manual_seed(123)
    # model = DummyGPTModel(GPT_CONFIG_124M)
    # logits = model(batch)
    # print("Output shape:", logits.shape)
    # print(logits)
    GPT_CONFIG_124M = json.load(open('../GPT_CONFIG_124M.json', mode='r', encoding='UTF-8'))

    print('----torch----')
    # ffn = FeedForward(GPT_CONFIG_124M)
    # x = torch.rand(2, 3, 768)
    # out = ffn(x)
    # print(out.shape)

    x = torch.rand(2, 4, 768)
    block = GPTArchitecture.TransformerBlock(GPT_CONFIG_124M)
    blockOut = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", blockOut.shape)

    model = GPTArchitecture.GPTModel(config=GPT_CONFIG_124M)
    out = model(batch)  # (2, 4)
    print(f'out shape: {out.shape}')
    total_para = sum(p.numel() for p in model.parameters())
    print(f'total parameter: {total_para}')

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_heads.weight.shape)

    total_params_gpt2 = (
        total_para - sum(p.numel()
                           ) for p in model.out_heads.parameters())
    print(f"Number of trainable parameters "
          f"considering weight tying: {total_params_gpt2}"
          )

    print('----tf----')
    tensor = tf.random.normal(shape=(2, 4, 768))
    # ffn_tf = GPTArchitectureTF.FeedForward(config=GPT_CONFIG_124M)
    # out = ffn_tf(tensor)
    # print(out.shape)
    blockTF = GPTArchitectureTF.TransformerBlock(GPT_CONFIG_124M)
    blockOutTF = blockTF(tensor)
    print("Input shape:", tensor.shape)
    print("Output shape:", blockOutTF.shape)
    # print('out"\n', out)

    modelTF = GPTArchitectureTF.GPTModel(GPT_CONFIG_124M)

    out_tensor = modelTF.call(batchTF)
    print(out_tensor.shape)

    # predict
    start_context = "Hello, I am"  # 你好，我是StNation Chicnets='.''.
    encoded = tokenizer.encode(start_context)
    print(f'encoded: {encoded}')
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)