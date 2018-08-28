import torch

from . import common
from . import decoder
from . import embedding


class AbstractDiscreteSequenceDecoder(common.Module):
    def __init__(self, vocab_size, word_dim, hidden_dim):
        super(AbstractDiscreteSequenceDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim

    def forward_loss(self, z, x, lens=None):
        raise NotImplementedError()

    def decode(self, z, bos_idx, eos_idx=None, max_len=100):
        raise NotImplementedError()


class DiscreteSequenceDecoder(AbstractDiscreteSequenceDecoder):
    name = "discrete-sequence-decoder"

    def __init__(self, *args,
                 embed=embedding.AbstractEmbedding,
                 decoder=decoder.AbstractSequenceDecoder, **kwargs):
        super(DiscreteSequenceDecoder, self).__init__(*args, **kwargs)
        self.embed_cls = embed
        self.decoder_cls = decoder
        self.embed = self.embed_cls(
            vocab_size=self.vocab_size,
            dim=self.word_dim
        )
        self.decoder = self.decoder_cls(
            in_dim=self.word_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.word_dim
        )

    def apply_output_embed(self, o):
        batch_size, seq_len, word_dim = o.size()
        weight = self.embed.weight.t()
        o = torch.mm(o.view(-1, word_dim), weight)
        return o.view(batch_size, seq_len, -1)

    def forward_loss(self, z, x, lens=None):
        x = self.invoke(self.embed, x)
        o = self.invoke(self.decoder, z, x, lens)
        return self.apply_output_embed(o)

    def decode(self, z, bos_idx, eos_idx=None, max_len=100):
        batch_size = z.size(0)
        x = z.new(batch_size, 1).long().fill_(bos_idx)
        has_eos = x.new(batch_size).zero_().byte()
        lens = x.new(batch_size).fill_(x.size(1)).long()
        while has_eos.prod().item() != 1 and lens.max() < max_len + 1:
            x_emb = self.invoke(self.embed, x)
            o = self.invoke(self.decoder, z, x_emb, lens)
            o = o[:, -1].unsqueeze(1)
            logits = self.apply_output_embed(o)
            logits = logits.squeeze(1)
            preds = logits.max(1)[1]
            x = torch.cat([x, preds.unsqueeze(1)], 1)
            has_eos = (preds == eos_idx) | has_eos
            lens += (1 - has_eos).long()
        return x, lens + 1


MODULES = [
    DiscreteSequenceDecoder
]