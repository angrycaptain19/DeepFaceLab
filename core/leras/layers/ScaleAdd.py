from core.leras import nn
tf = nn.tf

class ScaleAdd(nn.LayerBase):
    def __init__(self, ch, dtype=None, **kwargs):
        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype
        self.ch = ch

        super().__init__(**kwargs)

    def build_weights(self):
        self.weight = tf.get_variable("weight",(self.ch,), dtype=self.dtype, initializer=tf.initializers.zeros() )

    def get_weights(self):
        return [self.weight]

    def forward(self, inputs):
        shape = (1, 1, 1, self.ch) if nn.data_format == "NHWC" else (1, self.ch, 1, 1)
        weight = tf.reshape ( self.weight, shape )

        x0, x1 = inputs
        return x0 + x1*weight
nn.ScaleAdd = ScaleAdd