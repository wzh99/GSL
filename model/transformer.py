from tvm import relay

from gsl import Workload
from .common import batch_size

seq_len = 16
num_feat = 64
input_shape = (seq_len, num_feat)
batch_shape = (batch_size,) + input_shape


def get_workload(num_layers: int, d_model: int, num_heads: int, d_ff: int) -> Workload:
    return _TransformerCreator(num_layers, d_model, num_heads, d_ff).create()


class _TransformerCreator:
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_per_head = d_model // num_heads
        self.d_ff = d_ff

    def create(self) -> Workload:
        name = 'transformer_{}_{}_{}_{}'.format(self.num_layers, self.d_model,
                                                self.num_heads, self.d_ff)
        src = relay.var('src', shape=batch_shape)
        tgt = relay.var('tgt', shape=batch_shape)
        enc = self._encoder(src)
        dec = self._decoder(tgt, enc)
        return Workload.from_expr(dec, {src.name_hint, tgt.name_hint}, name=name)

    def _encoder(self, x: relay.Expr) -> relay.Expr:
        for i in range(self.num_layers):
            x = self._encoder_layer(x, 'e{}'.format(i))
        return x

    def _encoder_layer(self, x: relay.Expr, name: str) -> relay.Expr:
        mha = self._multi_head(x, x, x, name + '_mha')
        ln1 = self._layer_norm(mha + x, name + '_ln1')
        ff = self._feed_forward(ln1, name + '_ff')
        return self._layer_norm(ff + ln1, name + '_ln2')

    def _decoder(self, x: relay.Expr, enc: relay.Expr) -> relay.Expr:
        for i in range(self.num_layers):
            x = self._decoder_layer(x, enc, 'd{}'.format(i))
        return x

    def _decoder_layer(self, x: relay.Expr, enc: relay.Expr, name: str) -> relay.Expr:
        mha1 = self._multi_head(x, x, x, name + '_mha1')
        ln1 = self._layer_norm(mha1 + x, name + '_ln1')
        mha2 = self._multi_head(ln1, enc, enc, name + '_mha2')
        ln2 = self._layer_norm(mha2 + ln1, name + '_ln2')
        ff = self._feed_forward(ln2, name + '_ff')
        ln3 = self._layer_norm(ff + ln2, name + '_ln3')
        return ln3

    def _multi_head(self, q: relay.Expr, k: relay.Expr, v: relay.Expr, name: str) -> relay.Expr:
        q = self._split_heads(self._dense(q, self.d_model, name + '_dq'))
        k = self._split_heads(self._dense(k, self.d_model, name + '_dk'))
        v = self._split_heads(self._dense(v, self.d_model, name + '_dv'))
        att = relay.transpose(self._attention(q, k, v), axes=(0, 2, 1, 3))
        att = relay.reshape(att, (batch_size, seq_len, self.d_model))
        out = self._dense(att, self.d_model, name + '_d')
        return out

    def _split_heads(self, x: relay.Expr) -> relay.Expr:
        reshape = relay.reshape(x, (batch_size, seq_len, self.num_heads, self.d_per_head))
        trans = relay.transpose(reshape, axes=(0, 2, 1, 3))
        return trans  # (b, h, l, d)

    def _attention(self, q: relay.Expr, k: relay.Expr, v: relay.Expr) -> relay.Expr:
        q = relay.reshape(q, (-1, seq_len, self.d_per_head))
        k = relay.reshape(k, (-1, seq_len, self.d_per_head))
        logits = relay.nn.batch_matmul(q, k) / relay.sqrt(relay.const(float(self.d_per_head)))
        att_wts = relay.nn.softmax(logits)
        v = relay.reshape(v, (-1, seq_len, self.d_per_head))
        v = relay.transpose(v, axes=(0, 2, 1))
        matmul = relay.nn.batch_matmul(att_wts, v)
        return relay.reshape(matmul, (batch_size, self.num_heads, seq_len, self.d_per_head))

    def _feed_forward(self, x: relay.Expr, name: str) -> relay.Expr:
        d1 = self._dense(x, self.d_ff, name + '_d1')
        relu = relay.nn.relu(d1)
        d2 = self._dense(relu, self.d_model, name + '_d2')
        return d2

    @staticmethod
    def _layer_norm(x: relay.Expr, name: str) -> relay.Expr:
        gamma = relay.var(name + '_gamma')
        beta = relay.var(name + '_beta')
        ln = relay.nn.layer_norm(x, gamma, beta)
        return ln

    @staticmethod
    def _dense(x: relay.Expr, units: int, name: str, use_bias: bool = True) -> relay.Expr:
        wt = relay.var(name + '_wt')
        dense = relay.nn.dense(x, wt, units=units)
        if use_bias:
            bias = relay.var(name + '_bias')
            dense = relay.nn.bias_add(dense, bias, axis=-1)
        return dense
