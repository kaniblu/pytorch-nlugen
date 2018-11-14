from . import rnn
from . import common
from . import pooling
from . import nonlinear


class AbstractSequenceEncoder(common.Module):

    def __init__(self, in_dim, out_dim):
        super(AbstractSequenceEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim  

    def forward_loss(self, x, lens=None):
        raise NotImplementedError()


class RNNEncoder(AbstractSequenceEncoder):

    def __init__(self, *args, cell=rnn.GRUCell, **kwargs):
        super(RNNEncoder, self).__init__(*args, **kwargs)
        self.rnn_cls = cell
        self.nonlinear = nonlinear.get_default()(
            in_dim=self.in_dim,
            out_dim=self.in_dim
        )
        self.rnn = cell(
            input_dim=self.in_dim,
            hidden_dim=self.out_dim
        )


class LastStateRNNEncoder(RNNEncoder):

    name = "last-state-rnn-encoder"

    def forward_loss(self, x, lens=None):
        x = self.invoke(self.nonlinear, x)
        o, c, h = self.invoke(self.rnn, x, lens)
        return h


class PooledRNNEncoder(RNNEncoder):

    name = "pooled-rnn-encoder"

    def __init__(self, *args, pool=pooling.MaxPooling, **kwargs):
        super(PooledRNNEncoder, self).__init__(*args, **kwargs)
        self.pool_cls = pool
        self.pool = pool(self.out_dim)

    def forward_loss(self, x, lens=None):
        o, _, _ = self.invoke(self.rnn, x, lens)
        return self.invoke(self.pool, o, lens)


MODULES = [
    LastStateRNNEncoder,
    PooledRNNEncoder
]