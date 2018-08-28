from . import common
from . import fusion
from . import encoder
from . import embedding


class AbstractDiscreteSequenceEncoder(common.Module):
    """
    x: [batch_size, seq_len] LongTensor
    lens: [batch_size] LongTensor
    """
    def __init__(self, vocab_size, word_dim, hidden_dim):
        super(AbstractDiscreteSequenceEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim

    def forward_loss(self, x, lens=None):
        raise NotImplementedError()


class DiscreteSequenceEncoder(AbstractDiscreteSequenceEncoder):
    name = "discrete-sequence-encoder"

    def __init__(self, *args,
                 embed=embedding.AbstractEmbedding,
                 encoder=encoder.AbstractSequenceEncoder, **kwargs):
        super(DiscreteSequenceEncoder, self).__init__(*args, **kwargs)
        self.embed_cls = embed
        self.encoder = encoder
        self.embed = embed(
            vocab_size=self.vocab_size,
            dim=self.word_dim
        )
        self.encoder = encoder(
            in_dim=self.word_dim,
            out_dim=self.hidden_dim
        )

    def forward_loss(self, x, lens=None):
        x = self.invoke(self.embed, x)
        return self.invoke(self.encoder, x, lens)


MODULES = [
    DiscreteSequenceEncoder
]