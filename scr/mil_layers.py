from keras import layers, ops, regularizers, initializers
import keras




class MILAttentionLayer(layers.Layer):



    def __init__(self, weight_params_dim=256, kernel_initializer="glorot_uniform", kernel_regularizer=None, use_gated=True, **kwargs):
        super().__init__(**kwargs)
        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer
        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer


    def build(self, input_shape):
        input_dim = input_shape[0][1]
        self.v_weight_params = self.add_weight(
        shape=(input_dim, self.weight_params_dim), initializer=self.v_init, name="v", regularizer=self.v_regularizer, trainable=True)
        self.w_weight_params = self.add_weight(
        shape=(self.weight_params_dim, 1), initializer=self.w_init, name="w", regularizer=self.w_regularizer, trainable=True)
        if self.use_gated:
            self.u_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim), initializer=self.u_init, name="u", regularizer=self.u_regularizer, trainable=True)
        else:
            self.u_weight_params = None


    def call(self, inputs):
        scores = [self._score(x) for x in inputs] # list of [B,1]
        scores = ops.stack(scores) # [N_inst, B, 1]
        alpha = ops.softmax(scores, axis=0) # softmax over instances
        return [alpha[i] for i in range(alpha.shape[0])]


    def _score(self, x):
        orig = x
        x = ops.tanh(ops.tensordot(x, self.v_weight_params, axes=1))
        if self.use_gated:
            x = x * ops.sigmoid(ops.tensordot(orig, self.u_weight_params, axes=1))
        return ops.tensordot(x, self.w_weight_params, axes=1) # [B,1]
