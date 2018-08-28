"""Variational Recurrent Encoder-Decoder"""
import collections

import torch

from . import common
from . import encoder
from . import decoder
from . import nonlinear
from . import embedding


class DiscreteSequenceEncoder(common.Module):
    def __init__(self, vocab_size, word_dim, hidden_dim, *,
                 emb=embedding.AbstractEmbedding,
                 enc=encoder.AbstractSequenceEncoder):
        super(DiscreteSequenceEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.emb = emb
        self.enc = enc
        self.embedding = emb(
            vocab_size=vocab_size,
            dim=word_dim
        )
        self.encoder = enc(
            in_dim=self.word_dim,
            out_dim=self.hidden_dim
        )

    def forward_loss(self, x, lens):
        x = self.invoke(self.embedding, x)
        return self.invoke(self.encoder, x, lens)



class VariationalRecurrentEncoderDecoder(common.Module):
    name = "variational-recurrent-encoder-decoder"

    def __init__(self, z_dim, enc_word_dim, enc_vocab_size,
                 dec_word_dims, dec_vocab_sizes, kld_scale=1.0,
                 enc_emb_class=embedding.BasicEmbedding,
                 dec_emb_classes=(embedding.BasicEmbedding, ),
                 enc_class=encoder.AbstractSequenceEncoder,
                 dec_classes=(decoder.AbstractSequenceDecoder, )):
        super(VariationalRecurrentEncoderDecoder, self).__init__()
        self.z_dim = z_dim
        self.enc_word_dim, self.enc_vocab_size = enc_word_dim, enc_vocab_size
        self.dec_word_dims = dec_word_dims
        self.dec_vocab_sizes = dec_vocab_sizes
        self.kld_scale = kld_scale
        self.enc_emb_class = enc_emb_class
        self.dec_emb_classes = dec_emb_classes
        self.enc_class = enc_class
        self.dec_classes = dec_classes
        self.enc_emb = enc_emb_class(
            vocab_size=enc_vocab_size,
            dim=enc_word_dim,
        )
        self.dec_embs = common.ModuleList(
            modules=[emb_cls(
                vocab_size=vocab_size,
                dim=word_dim
            ) for emb_cls, vocab_size, word_dim in
                zip(self.dec_emb_classes, dec_vocab_sizes, dec_word_dims)]
        )
        self.mu_linear = common.Linear(
            in_features=z_dim,
            out_features=z_dim
        )
        self.logvar_linear = common.Linear(
            in_features=z_dim,
            out_features=z_dim
        )
        self.encoder = enc_class(
            in_dim=enc_word_dim,
            hidden_dim=z_dim
        )
        self.decoders = common.ModuleList([dec_cls(
            in_dim=word_dim,
            hidden_dim=z_dim,
            out_dim=word_dim,
        ) for dec_cls, word_dim in zip(dec_classes, dec_word_dims)])
        self.z_nonlinears = common.ModuleList([
            nonlinear.get_default()(z_dim, z_dim)
            for _ in range(len(dec_word_dims))
        ])

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        rnd = torch.randn_like(std)
        return rnd * std + mu

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)

    def apply_output_embed(self, emb, o):
        batch_size, seq_len, word_dim = o.size()
        weight = emb.weight.t()
        o = torch.mm(o.view(-1, word_dim), weight)
        return o.view(batch_size, seq_len, -1)

    def forward_loss(self, x, ys, x_lens=None, ys_lens=None):
        x = self.invoke(self.enc_emb, x)
        h = self.invoke(self.encoder, x, x_lens)
        mu = self.invoke(self.mu_linear, h)
        logvar = self.invoke(self.logvar_linear, h)
        yield "loss", self.kld_loss(mu, logvar) * self.kld_scale
        z = self.sample(mu, logvar)
        zs = [self.invoke(layer, z) for layer in self.z_nonlinears]
        yield "pass", tuple(
            self.apply_output_embed(
                emb=emb,
                o=self.invoke(decoder, z, self.invoke(emb, y), lens)
            ) for emb, decoder, z, y, lens in \
            zip(self.dec_embs, self.decoders, zs, ys, ys_lens)
        )

    def decode_single(self, dec, emb, z, bos_idx, eos_idx=None, max_len=100):
        batch_size = z.size(0)
        x = z.new(batch_size, 1).long().fill_(bos_idx)
        has_eos = x.new(batch_size).zero_().byte()
        lens = x.new(batch_size).fill_(x.size(1)).long()
        while has_eos.prod().item() != 1 and lens.max() < max_len + 1:
            x_emb = self.invoke(emb, x)
            o = self.invoke(dec, z, x_emb, lens)
            o = o[:, -1].unsqueeze(1)
            logits = self.apply_output_embed(emb, o)
            logits = logits.squeeze(1)
            preds = logits.max(1)[1]
            x = torch.cat([x, preds.unsqueeze(1)], 1)
            has_eos = (preds == eos_idx) | has_eos
            lens += (1 - has_eos).long()
        return x, lens + 1

    def decode(self, z, bos_idxs, eos_idxs=None, max_len=100):
        batch_size = z.size(0)
        if not isinstance(eos_idxs, collections.Sequence):
            eos_idxs = [eos_idxs] * batch_size
        if not isinstance(bos_idxs, collections.Sequence):
            bos_idxs = [bos_idxs] * batch_size
        return [self.decode_single(
            dec=decoder,
            emb=emb,
            z=z,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            max_len=max_len
        ) for decoder, emb, bos_idx, eos_idx in \
            zip(self.decoders, self.dec_embs, bos_idxs, eos_idxs)]


MODULES = [
    VariationalRecurrentEncoderDecoder
]