from keras import layers, Model, regularizers
from .mil_layers import MILAttentionLayer




def build_model(instance_shape, num_classes, dense1=128, dense2=64, att_dim=256, gated=True, l2=0.01, dropout=0.0):
    inputs, embeddings = [], []
    shared1 = layers.Dense(dense1, activation="relu")
    shared2 = layers.Dense(dense2, activation="relu")
    if dropout > 0:
        drop = layers.Dropout(dropout)
    for _ in range(instance_shape[0]):
        raise ValueError("instance_shape should be like (n_features,) when building per-instance stream")




def build_mil(instance_shape, num_classes, dense1=128, dense2=64, att_dim=256, gated=True):
    inputs, embeddings = [], []
    shared_dense_layer_1 = layers.Dense(dense1, activation="relu")
    shared_dense_layer_2 = layers.Dense(dense2, activation="relu")
    for _ in range(3): # will be dynamically replaced by BAG_SIZE at call site
        inp = layers.Input(instance_shape)
        flat = layers.Flatten()(inp)
        d1 = shared_dense_layer_1(flat)
        d2 = shared_dense_layer_2(d1)
        inputs.append(inp); embeddings.append(d2)


    alpha = MILAttentionLayer(weight_params_dim=att_dim, kernel_regularizer=regularizers.L2(0.01), use_gated=gated, name="alpha")(embeddings)
    multiplied = [layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))]
    concat = layers.concatenate(multiplied, axis=1)
    output = layers.Dense(num_classes, activation="softmax")(concat)
    return Model(inputs, output)