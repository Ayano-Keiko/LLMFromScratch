import tensorflow as tf
import keras
import torch


def text_to_id(text: str, tokenizer, model):
    tokens = tokenizer.encode(text)
    if isinstance(model, (tf.Module, keras.Model)):
        token_tensor = tf.constant(tokens)
        return tf.expand_dims(token_tensor, axis=0)
    elif isinstance(model, torch.nn.Module):
        token_tensor = torch.tensor(tokens)
        return torch.unsqueeze(token_tensor, dim=0)
    else:
        return [tokens]

def id_to_text(tokenIDs, tokenizer, model) -> str:
    if isinstance(model, (tf.Module, keras.Model)):
        return tokenizer.decode( tf.squeeze(tokenIDs, axis=0) )
    elif isinstance(model, torch.nn.Module):
        return tokenizer.decode(tokenIDs.squeeze(0).tolist())
    else:
        raise tokenizer.decode(tokenIDs[0])

