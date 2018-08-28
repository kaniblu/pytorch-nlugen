import torch

from . import common
from . import decoder
from . import embedding


class AbstractPairedDiscreteDecoders(common.Module):
    """decoder0 -> decoder1"""
    def __init__(self, hidden_dim, vocab_size0, word_dim0,
                 vocab_size1, word_dim1):
        super(AbstractPairedDiscreteDecoders, self).__init__()
        self.vocab_sizes = [vocab_size0, vocab_size1]
        self.word_dims = [word_dim0, word_dim1]
        self.hidden_dim = hidden_dim

    def forward_loss(self, z, x, y, lens=None):
        raise NotImplementedError()

    def decode(self, z, x_bos, y_bos, x_eos=None, max_len=100):
        raise NotImplementedError()


class PairedDiscreteDecoders(AbstractPairedDiscreteDecoders):
    name = "paired-discrete-decoders"

    def __init__(self, *args,
                 embed0=embedding.AbstractEmbedding,
                 embed1=embedding.AbstractEmbedding,
                 decoder0=decoder.AbstractSequenceDecoder,
                 decoder1=decoder.AbstractSequenceDecoder, **kwargs):
        super(PairedDiscreteDecoders, self).__init__(*args, **kwargs)
        self.embed_classes = [embed0, embed1]
        self.decoder_classes = [decoder0, decoder1]
        self.embeds = common.ModuleList([cls(
            vocab_size=self.vocab_sizes[i],
            dim=self.word_dims[i]
        ) for i, cls in enumerate(self.embed_classes)])
        self.decoders = common.ModuleList([
            self.decoder_classes[0](
                in_dim=self.word_dims[0],
                hidden_dim=self.hidden_dim,
                out_dim=self.word_dims[0]
            ),
            self.decoder_classes[1](
                in_dim=self.word_dims[1] + self.word_dims[0],
                hidden_dim=self.hidden_dim,
                out_dim=self.word_dims[1]
            )
        ])

    @staticmethod
    def apply_output_embed(embed, o):
        batch_size, seq_len, word_dim = o.size()
        weight = embed.weight.t()
        o = torch.mm(o.view(-1, word_dim), weight)
        return o.view(batch_size, seq_len, -1)

    def forward_loss(self, z, x, y, lens=None):
        xs = [self.invoke(embed, x) for x, embed in zip((x, y), self.embeds)]
        x, y = xs
        x_o = self.invoke(self.decoders[0], z, x, lens)
        y = torch.cat([x_o, y], 2)
        y_o = self.invoke(self.decoders[1], z, y, lens)
        return tuple(self.apply_output_embed(embed, o)
                     for o, embed in zip((x_o, y_o), self.embeds))

    def decode_single(self, dec, emb, z, bos, eos=None, max_len=100, inp=None):
        batch_size = z.size(0)
        x = z.new(batch_size, 1).long().fill_(bos)
        has_eos = x.new(batch_size).zero_().byte()
        lens = x.new(batch_size).fill_(x.size(1)).long()
        while has_eos.prod().item() != 1 and lens.max() < max_len + 1:
            x_emb = self.invoke(emb, x)
            if inp is not None:
                x_emb = torch.cat([inp[:, :x_emb.size(1)], x_emb], 2)
            o = self.invoke(dec, z, x_emb, lens)
            o = o[:, -1].unsqueeze(1)
            logits = self.apply_output_embed(emb, o)
            logits = logits.squeeze(1)
            preds = logits.max(1)[1]
            x = torch.cat([x, preds.unsqueeze(1)], 1)
            if eos is not None:
                has_eos = (preds == eos) | has_eos
            lens += (1 - has_eos).long()
        return x, lens + 1

    def decode(self, z, x_bos, y_bos, x_eos=None, max_len=100):
        x, x_lens = self.decode_single(
            dec=self.decoders[0],
            emb=self.embeds[0],
            z=z,
            bos=x_bos,
            eos=x_eos,
            max_len=max_len
        )
        x_emb = self.invoke(self.embeds[0], x)
        y, y_lens = self.decode_single(
            dec=self.decoders[1],
            emb=self.embeds[1],
            z=z,
            bos=y_bos,
            max_len=x_emb.size(1),
            inp=x_emb
        )
        return x, y, x_lens