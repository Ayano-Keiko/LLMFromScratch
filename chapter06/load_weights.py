import tensorflow as tf
import numpy
import tensorflow.compat.v1 as tf1

def print_checkpoint(save_path):
    reader = tf.train.load_checkpoint(save_path)
    shapes = reader.get_variable_to_shape_map()
    dtypes = reader.get_variable_to_dtype_map()
    print(f"Checkpoint at '{save_path}':")
    for key in shapes:
        print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
          f"value={reader.get_tensor(key)})")

def load_weights_into_gpt(model, params):
    model.pos_emb.set_weights([params['wpe']])
    model.tok_emb.set_weights([params['wte']])

    for b in range(len(params["blocks"])):
        block = model.transformers[b]

        q_w, k_w, v_w = numpy.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        q_b, k_b, v_b = numpy.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)

        # MultiHeadAttention weights
        block.att.query.set_weights([q_w, q_b])
        block.att.key.set_weights([k_w, k_b])
        block.att.value.set_weights([v_w, v_b])
        # Attention output projection
        block.att.out_proj.set_weights([
            params["blocks"][b]["attn"]["c_proj"]["w"],
            params["blocks"][b]["attn"]["c_proj"]["b"]
        ])
        # FeedForward (MLP) weights
        # Your FeedForward class uses self.linear1 and self.linear2
        block.ff.linear1.set_weights([
            params["blocks"][b]["mlp"]["c_fc"]["w"],
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        ])
        block.ff.linear2.set_weights([
            params["blocks"][b]["mlp"]["c_proj"]["w"],
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        ])
        # Layer Normalization (Scale/Shift)
        block.norm1.set_weights([
            params["blocks"][b]["ln_1"]["g"],
            params["blocks"][b]["ln_1"]["b"]
        ])
        block.norm2.set_weights([
            params["blocks"][b]["ln_2"]["g"],
            params["blocks"][b]["ln_2"]["b"]
        ])

    model.final_norm.set_weights([
        params["g"], params["b"]
    ])
    model.out_head.set_weights([
        params["wte"].T, numpy.zeros(model.cfg["vocab_size"])
    ])


