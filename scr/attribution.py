import numpy as np
import keras




def get_attention_weights(model, data):
    inter = keras.Model(model.input, model.get_layer("alpha").output)
    alpha_list = inter.predict(data, verbose=0)
    # alpha_list is list length = BAG_SIZE of [B,1]; squeeze and swap back to [B, BAG_SIZE]
    return np.squeeze(np.swapaxes(alpha_list, 1, 0))




def attention_weighted_instance_probs(attn, bag_probs):
# attn: [B, N]; bag_probs: [B, C]
# if needed per-instance attribution to class probs, multiply instance attn by bag-level prob vector
    return attn[..., None] * bag_probs[:, None, :]
