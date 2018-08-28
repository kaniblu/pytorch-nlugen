import torch
import numpy as np
import torch.nn as nn

import utils
from . import common
from . import fusion
from . import gaussian
from . import nonlinear
from . import embedding
from . import encoder
from . import decoder


class AbstractLanguageUnderstandingVAE(common.Module):
    def __init__(self, z_dim, word_dim, label_dim, intent_dim,
                 num_words, num_labels, num_intents):
        super(AbstractLanguageUnderstandingVAE, self).__init__()
        self.z_dim = z_dim
        self.word_dim = word_dim
        self.label_dim = label_dim
        self.intent_dim = intent_dim
        self.num_words = num_words
        self.num_labels = num_labels
        self.num_intents = num_intents
        self.word_dims = [word_dim, label_dim, intent_dim]
        self.vocab_sizes = [num_words, num_labels, num_intents]

    def embeddings(self):
        raise NotImplementedError()

    def forward_loss(self, words, labels, intents, lens):
        raise NotImplementedError()

    def generate(self, z, word_bos, label_bos, word_eos=None, max_len=100):
        raise NotImplementedError()

    def encode(self, words, labels, intents, lens):
        """
        returns a tuple of means and stds
        :param words:
        :param labels:
        :param intents:
        :param lens:
        :return: [batch_size x z_dim], [batch_size x z_dim]
        """
        raise NotImplementedError()


class LanguageUnderstandingVAE(AbstractLanguageUnderstandingVAE):
    name = "luvae"

    def __init__(self, *args, singulars=False,
                 sampling=gaussian.AbstractGaussianSampling,
                 word_embed=embedding.AbstractEmbedding,
                 label_embed=embedding.AbstractEmbedding,
                 intent_embed=embedding.AbstractEmbedding,
                 word_encoder=encoder.AbstractSequenceEncoder,
                 label_encoder=encoder.AbstractSequenceEncoder,
                 intent_encoder=nonlinear.get_default(),
                 word_decoder=decoder.AbstractSequenceDecoder,
                 label_decoder=decoder.AbstractSequenceDecoder,
                 intent_decoder=nonlinear.get_default(),
                 fusion=fusion.BaseFusion, **kwargs):
        super(LanguageUnderstandingVAE, self).__init__(*args, **kwargs)
        self.singulars = singulars
        self.sampling_cls = sampling
        self.word_embed_cls = word_embed
        self.label_embed_cls = label_embed
        self.intent_embed_cls = intent_embed
        self.embed_classes = [word_embed, label_embed, intent_embed]
        self.word_encoder_cls = word_encoder
        self.label_encoder_cls = label_encoder
        self.intent_encoder_cls = intent_encoder
        self.word_decoder_cls = word_decoder
        self.label_decoder_cls = label_decoder
        self.intent_decoder_cls = intent_decoder
        self.fusion_cls = fusion
        self.embeds = common.ModuleList([embed_cls(
            vocab_size=vocab_size,
            dim=dim
        ) for embed_cls, vocab_size, dim in
            zip(self.embed_classes, self.vocab_sizes, self.word_dims)])

        encoder_word = self.word_encoder_cls(
            in_dim=self.word_dim,
            out_dim=self.z_dim
        )
        encoder_label = self.label_encoder_cls(
            in_dim=self.label_dim,
            out_dim=self.z_dim
        )
        encoder_intent = self.intent_encoder_cls(
            in_dim=self.intent_dim,
            out_dim=self.z_dim
        )
        self.encoders = common.ModuleList([
            encoder_word,
            encoder_label,
            encoder_intent
        ])

        decoder_word = self.word_decoder_cls(
            in_dim=self.word_dim,
            hidden_dim=self.z_dim,
            out_dim=self.word_dim
        )
        decoder_label = self.label_decoder_cls(
            in_dim=self.label_dim,
            hidden_dim=self.z_dim,
            out_dim=self.label_dim
        )
        decoder_intent = self.intent_decoder_cls(
            in_dim=self.z_dim,
            out_dim=self.intent_dim
        )
        self.decoders = common.ModuleList([
            decoder_word,
            decoder_label,
            decoder_intent
        ])
        self.fusion = self.fusion_cls(
            in_dims=(self.z_dim, ) * 3,
            out_dim=self.z_dim
        )
        self.sampling = self.sampling_cls(
            in_dim=self.z_dim,
            out_dim=self.z_dim
        )
        self.sampling_s = None
        if self.singulars:
            self.sampling_s = common.ModuleList([self.sampling_cls(
                in_dim=self.z_dim,
                out_dim=self.z_dim
            ) for _ in range(3)])

    @staticmethod
    def apply_output_embed(embed, o):
        o_size = o.size()
        weight = embed.weight.t()
        o = torch.mm(o.view(-1, o_size[-1]), weight)
        return o.view(*o_size[:-1], -1)

    def forward_loss(self, w, l, i, lens):
        w = self.invoke(self.embeds[0], w)
        l = self.invoke(self.embeds[1], l)
        i = self.invoke(self.embeds[2], i)
        hs = (
            self.invoke(self.encoders[0], w, lens),
            self.invoke(self.encoders[1], l, lens),
            self.invoke(self.encoders[2], i)
        )
        h = self.invoke(self.fusion, *hs)
        z = self.invoke(self.sampling, h)
        (w, l), lens = [x[:, :-1].contiguous() for x in (w, l)], lens - 1
        os = (
            self.invoke(self.decoders[0], z, w, lens),
            self.invoke(self.decoders[1], z, l, lens),
            self.invoke(self.decoders[2], z)
        )
        logits = tuple(self.apply_output_embed(embed, o)
                       for embed, o in zip(self.embeds, os))
        if self.singulars:
            zs = [self.invoke(self.sampling, h)
                  for sampling, h in zip(self.sampling_s, hs)]
            os_s = (
                self.invoke(self.decoders[0], zs[0], w, lens),
                self.invoke(self.decoders[1], zs[1], l, lens),
                self.invoke(self.decoders[2], zs[2])
            )
            logits_s = tuple(self.apply_output_embed(embed, o)
                             for embed, o in zip(self.embeds, os_s))
            yield "pass", (logits, logits_s)
        else:
            yield "pass", logits

    def decode_single(self, dec, emb, z, bos, eos=None, max_len=100):
        batch_size = z.size(0)
        x = z.new(batch_size, 1).long().fill_(bos)
        has_eos = x.new(batch_size).zero_().byte()
        lens = x.new(batch_size).fill_(x.size(1)).long()
        while has_eos.prod().item() != 1 and lens.max() < max_len:
            x_emb = self.invoke(emb, x)
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

    def decode_single_beamsearch(self, dec, emb, z, bos,
                                 eos=None, max_len=100, beam_size=3):
        max_float = 1e10
        min_float = -1e10
        softmax = nn.Softmax(2)
        batch_size = z.size(0)
        x = z.new(batch_size, beam_size, 1).long().fill_(bos)
        has_eos = x.new(batch_size, beam_size).zero_().byte()
        probs = z.new(batch_size, beam_size).fill_(1.0)
        lens = x.new(batch_size, beam_size).fill_(1).long()
        while has_eos.prod().item() != 1 and lens.max() < max_len:
            x = x.view(batch_size * beam_size, -1)
            x_emb = self.invoke(emb, x)
            z_exp = z.unsqueeze(1).expand(batch_size, beam_size, -1) \
                .contiguous().view(batch_size * beam_size, -1)
            o = self.invoke(dec, z_exp, x_emb, lens.view(-1))[:, -1]
            x = x.view(batch_size, beam_size, -1)
            o = o.view(batch_size, beam_size, -1)
            logits = self.apply_output_embed(emb, o)
            # for beams that already generated <eos>, no more probability
            # depreciation.
            if eos is not None:
                eos_mask = has_eos.unsqueeze(-1).float()
                logits_eos = torch.full_like(logits, min_float)
                logits_eos[:, :, eos] = max_float
                logits = logits * (1 - eos_mask) + logits_eos * eos_mask
            # batch_size x beam_size x vocab_size
            p_vocab = probs.unsqueeze(-1) * softmax(logits)
            vocab_size = p_vocab.size(-1)
            # we utilize indices
            probs, idx = torch.sort(p_vocab.view(batch_size, -1), 1, True)
            probs, idx = probs[:, :beam_size], idx[:, :beam_size].long()
            beam_idx, preds = idx / vocab_size, idx % vocab_size
            x = torch.gather(x, 1, beam_idx.unsqueeze(-1).expand_as(x))
            x = torch.cat([x, preds.unsqueeze(-1)], 2)
            if eos is not None:
                has_eos = (preds == eos) | has_eos
            lens += (1 - has_eos).long()
        return x[:, 0], lens[:, 0] + 1

    def generate(self, z, word_bos, label_bos, word_eos=None, max_len=100,
                 beam_search=False, beam_size=3):
        if beam_search:
            decode_fn = self.decode_single
        else:
            decode_fn = lambda *args, **kwargs: \
                self.decode_single_beamsearch(*args, beam_size=beam_size, **kwargs)
        w, lens = decode_fn(
            dec=self.decoders[0],
            emb=self.embeds[0],
            z=z,
            bos=word_bos,
            eos=word_eos,
            max_len=max_len
        )
        l, _ = decode_fn(
            dec=self.decoders[1],
            emb=self.embeds[1],
            z=z,
            bos=label_bos,
            max_len=w.size(1)
        )
        i = self.invoke(self.decoders[2], z)
        i = self.apply_output_embed(self.embeds[2], i).max(1)[1]
        return w, l, i, lens

    def embeddings(self):
        for embed in self.embeds:
            yield embed


class ConditionalLUVAE(AbstractLanguageUnderstandingVAE):
    name = "cond-luvae"

    def __init__(self, *args, word_dropout=0.0,
                 embed_dropout=False, dropout_prob=0.5,
                 label_encoding=False, intent_encoding=False,
                 no_label_word_dropout=False, no_intent_word_dropout=False,
                 sampling=gaussian.AbstractGaussianSampling,
                 word_embed=embedding.AbstractEmbedding,
                 label_embed=embedding.AbstractEmbedding,
                 intent_embed=embedding.AbstractEmbedding,
                 word_encoder=encoder.AbstractSequenceEncoder,
                 label_encoder=encoder.AbstractSequenceEncoder,
                 intent_encoder=nonlinear.get_default(),
                 word_decoder=decoder.AbstractSequenceDecoder,
                 label_decoder=decoder.AbstractSequenceDecoder,
                 intent_decoder=encoder.AbstractSequenceEncoder,
                 fusion=fusion.BaseFusion, **kwargs):
        super(ConditionalLUVAE, self).__init__(*args, **kwargs)
        self.word_dropout = word_dropout
        self.label_word_dropout = not no_label_word_dropout
        self.intent_word_dropout = not no_intent_word_dropout
        self.embed_dropout = embed_dropout
        self.dropout_prob = dropout_prob
        self.label_encoding = label_encoding
        self.intent_encoding = intent_encoding
        self.sampling_cls = sampling
        self.word_embed_cls = word_embed
        self.label_embed_cls = label_embed
        self.intent_embed_cls = intent_embed
        self.embed_classes = [word_embed, label_embed, intent_embed]
        self.word_encoder_cls = word_encoder
        self.label_encoder_cls = label_encoder
        self.intent_encoder_cls = intent_encoder
        self.word_decoder_cls = word_decoder
        self.label_decoder_cls = label_decoder
        self.intent_decoder_cls = intent_decoder
        self.fusion_cls = fusion

        if self.embed_dropout:
            self.dropout = nn.Dropout(self.dropout_prob)
        self.embeds = common.ModuleList([embed_cls(
            vocab_size=vocab_size,
            dim=dim
        ) for embed_cls, vocab_size, dim in
            zip(self.embed_classes, self.vocab_sizes, self.word_dims)])

        encoder_word = self.word_encoder_cls(
            in_dim=self.word_dim,
            out_dim=self.z_dim
        )
        encoder_label = self.label_encoder_cls(
            in_dim=self.label_dim,
            out_dim=self.z_dim
        )
        encoder_intent = self.intent_encoder_cls(
            in_dim=self.intent_dim,
            out_dim=self.z_dim
        )
        self.encoders = common.ModuleList([
            encoder_word,
            encoder_label,
            encoder_intent
        ])

        decoder_word = self.word_decoder_cls(
            in_dim=self.word_dim,
            hidden_dim=self.z_dim,
            out_dim=self.word_dim
        )
        decoder_label = self.label_decoder_cls(
            in_dim=self.label_dim + self.word_dim,
            hidden_dim=self.z_dim,
            out_dim=self.label_dim
        )
        decoder_intent = self.intent_decoder_cls(
            in_dim=self.z_dim + self.word_dim,
            out_dim=self.intent_dim
        )
        self.decoders = common.ModuleList([
            decoder_word,
            decoder_label,
            decoder_intent
        ])
        self.fusion = self.fusion_cls(
            in_dims=(self.z_dim, ) * 3,
            out_dim=self.z_dim
        )
        self.sampling = self.sampling_cls(
            in_dim=self.z_dim,
            out_dim=self.z_dim
        )
        self.sampling_s = None

    @staticmethod
    def apply_output_embed(embed, o):
        o_size = o.size()
        weight = embed.weight.t()
        o = torch.mm(o.view(-1, o_size[-1]), weight)
        return o.view(*o_size[:-1], -1)

    @staticmethod
    def randidx(x, size):
        """x is either integer or array-like probability distribution"""
        if isinstance(x, int):
            return torch.randint(0, x, size)
        else:
            x = np.random.choice(np.arange(len(x)), p=x, size=size)
            return torch.tensor(x)

    def forward_loss(self, w, l, i, lens):
        w = self.invoke(self.embeds[0], w)
        l = self.invoke(self.embeds[1], l)
        i = self.invoke(self.embeds[2], i)
        if self.embed_dropout:
            w = self.dropout(w)
            l = self.dropout(l)
            i = self.dropout(i)

        hs = [self.invoke(self.encoders[0], w, lens)]
        if self.label_encoding:
            hs.append(self.invoke(self.encoders[1], l, lens))
        if self.intent_encoding:
            hs.append(self.invoke(self.encoders[2], i))
        h = self.invoke(self.fusion, *hs)
        z = self.invoke(self.sampling, h)
        if self.word_dropout:
            probs = (self.word_dropout, 1 - self.word_dropout)
            rand_mask = self.randidx(probs, w.size()[:2]).to(w.device).float()
            rand_mask = rand_mask.unsqueeze(-1)
            input_word = rand_mask * w + (1 - rand_mask) * torch.zeros_like(w)
            if self.intent_word_dropout:
                input_intent_w = input_word
            else:
                input_intent_w = w
            if self.label_word_dropout:
                input_label_w = input_word
            else:
                input_label_w = w
        else:
            input_word = w
            input_intent_w = w
            input_label_w = w
        input_intent = torch.cat([common.expand_match(z, w, 1),
                                  input_intent_w], 2)
        input_label = torch.cat([utils.roll_left(input_label_w), l], 2)
        os = (
            self.invoke(self.decoders[0], z, input_word, lens),
            self.invoke(self.decoders[1], z, input_label, lens),
            self.invoke(self.decoders[2], input_intent, lens)
        )
        logits = tuple(self.apply_output_embed(embed, o)
                       for embed, o in zip(self.embeds, os))
        yield "pass", logits

    def decode_single(self, dec, emb, z, bos, eos=None, cond=None, max_len=100):
        batch_size = z.size(0)
        x = z.new(batch_size, 1).long().fill_(bos)
        has_eos = x.new(batch_size).zero_().byte()
        lens = x.new(batch_size).fill_(x.size(1)).long()
        if cond is not None:
            max_len = min(cond.size(1), max_len)
        while has_eos.prod().item() != 1 and lens.max() < max_len:
            seq_len = x.size(1)
            x_emb = self.invoke(emb, x)
            if cond is not None:
                input = torch.cat([cond[:, :seq_len], x_emb], 2)
            else:
                input = x_emb
            o = self.invoke(dec, z, input, lens)
            o = o[:, -1].unsqueeze(1)
            logits = self.apply_output_embed(emb, o)
            logits = logits.squeeze(1)
            preds = logits.max(1)[1]
            x = torch.cat([x, preds.unsqueeze(1)], 1)
            if eos is not None:
                has_eos = (preds == eos) | has_eos
            lens += (1 - has_eos).long()
        return x, lens + 1

    def decode_single_beamsearch(self, dec, emb, z, bos, cond=None,
                                 eos=None, max_len=100, beam_size=3):
        max_float = 1e10
        min_float = -1e10
        softmax = nn.Softmax(2)
        batch_size = z.size(0)
        # forces the beam searcher to search from the first index only
        # in the beginning
        x = z.new(batch_size, 1, 1).long().fill_(bos)
        has_eos = x.new(batch_size, 1).zero_().byte()
        probs = z.new(batch_size, 1).fill_(1.0)
        lens = x.new(batch_size, 1).fill_(1).long()
        if cond is not None:
            max_len = min(cond.size(1), max_len)
        while has_eos.prod().item() != 1 and lens.max() < max_len:
            cur_beamsize, seq_len = x.size(1), x.size(2)
            x_emb = self.invoke(emb, x)
            if cond is not None:
                cond_exp = cond.unsqueeze(1).expand(-1, cur_beamsize, -1, -1)
                input = torch.cat([cond_exp[:, :, :seq_len], x_emb], 3)
            else:
                input = x_emb
            input = input.view(batch_size * cur_beamsize, seq_len, -1)
            z_exp = z.unsqueeze(1).expand(batch_size, cur_beamsize, -1) \
                .contiguous().view(batch_size * cur_beamsize, -1)
            o = self.invoke(dec, z_exp, input, lens.view(-1))[:, -1]
            x = x.view(batch_size, cur_beamsize, -1)
            o = o.view(batch_size, cur_beamsize, -1)
            logits = self.apply_output_embed(emb, o)
            # for beams that already generated <eos>, no more probability
            # depreciation.
            if eos is not None:
                eos_mask = has_eos.unsqueeze(-1).float()
                logits_eos = torch.full_like(logits, min_float)
                logits_eos[:, :, eos] = max_float
                logits = logits * (1 - eos_mask) + logits_eos * eos_mask
            # batch_size x beam_size x vocab_size
            p_vocab = probs.unsqueeze(-1) * softmax(logits)
            vocab_size = p_vocab.size(-1)
            # utilize 2d-flattened indices
            probs, idx = torch.sort(p_vocab.view(batch_size, -1), 1, True)
            probs, idx = probs[:, :beam_size], idx[:, :beam_size].long()
            beam_idx, preds = idx / vocab_size, idx % vocab_size
            x = torch.gather(x, 1, beam_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            x = torch.cat([x, preds.unsqueeze(-1)], 2)
            if eos is not None:
                has_eos = torch.gather(has_eos, 1, beam_idx)
                has_eos = (preds == eos) | has_eos
            lens = torch.gather(lens, 1, beam_idx)
            lens += (1 - has_eos).long()
        return x, lens + 1, probs

    def predict_intent(self, z, w, lens):
        softmax = nn.Softmax(1)
        x = torch.cat([common.expand_match(z, w, 1), w], 2)
        o = self.invoke(self.decoders[2], x, lens)
        probs = softmax(self.apply_output_embed(self.embeds[2], o))
        pi, i = probs.max(1)
        return i, pi

    def encode(self, words, labels, intents, lens):
        w = self.invoke(self.embeds[0], words)
        h = self.invoke(self.encoders[0], w, lens)
        h = self.invoke(self.fusion, h)
        return self.sampling.mean(h)

    def predict(self, w, lens, label_bos, beam_size=3):
        w = self.invoke(self.embeds[0], w)
        h = self.invoke(self.encoders[0], w, lens)
        h = self.invoke(self.fusion, h)
        z = self.sampling.mean(h)
        l, _, pl = self.decode_single_beamsearch(
            dec=self.decoders[1],
            emb=self.embeds[1],
            z=z,
            bos=label_bos,
            cond=utils.roll_left(w),
            beam_size=beam_size
        )
        l, pl = l[:, 0], pl[:, 0]
        i, pi = self.predict_intent(z, w, lens)
        return (l, i), (pl, pi)

    def generate(self, z, word_bos, label_bos, word_eos=None, max_len=100,
                 beam_size=3):
        batch_size = z.size(0)
        w, lens, pw = self.decode_single_beamsearch(
            dec=self.decoders[0],
            emb=self.embeds[0],
            z=z,
            bos=word_bos,
            eos=word_eos,
            max_len=max_len,
            beam_size=beam_size
        )
        w_flat = w.contiguous().view(batch_size * beam_size, -1)
        z_exp = z.unsqueeze(1).expand(batch_size, beam_size, -1).contiguous()
        z_exp = z_exp.view(batch_size * beam_size, -1)
        w_emb = self.invoke(self.embeds[0], w_flat)
        l, _, pl = self.decode_single_beamsearch(
            dec=self.decoders[1],
            emb=self.embeds[1],
            z=z_exp,
            bos=label_bos,
            max_len=w_flat.size(1),
            cond=utils.roll_left(w_emb),
            beam_size=beam_size
        )
        l = l[:, 0].view(batch_size, beam_size, -1)
        pl = pl[:, 0].view(batch_size, beam_size)
        i, pi = self.predict_intent(z_exp, w_emb, lens.view(-1))
        pi = pi.view(batch_size, beam_size)
        i = i.view(batch_size, beam_size)
        return (w, l, i, lens), (pw, pl, pi)

    def embeddings(self):
        for embed in self.embeds:
            yield embed


class NewConditionalLUVAE(AbstractLanguageUnderstandingVAE):
    name = "new-cond-luvae"

    def __init__(self, *args, word_dropout=0.0,
                 dropout=False, dropout_prob=0.5, batch_norm=False,
                 sampling=gaussian.AbstractGaussianSampling,
                 word_embed=embedding.AbstractEmbedding,
                 label_embed=embedding.AbstractEmbedding,
                 intent_embed=embedding.AbstractEmbedding,
                 word_encoder=encoder.AbstractSequenceEncoder,
                 label_encoder=encoder.AbstractSequenceEncoder,
                 intent_encoder=nonlinear.get_default(),
                 word_decoder=decoder.AbstractSequenceDecoder,
                 label_decoder=decoder.AbstractSequenceDecoder,
                 intent_decoder=encoder.AbstractSequenceEncoder, **kwargs):
        super(NewConditionalLUVAE, self).__init__(*args, **kwargs)
        self.word_dropout = word_dropout
        self.should_dropout = dropout
        self.should_batchnorm = batch_norm
        self.dropout_prob = dropout_prob
        self.sampling_cls = sampling
        self.word_embed_cls = word_embed
        self.label_embed_cls = label_embed
        self.intent_embed_cls = intent_embed
        self.embed_classes = [word_embed, label_embed, intent_embed]
        self.word_encoder_cls = word_encoder
        self.label_encoder_cls = label_encoder
        self.intent_encoder_cls = intent_encoder
        self.word_decoder_cls = word_decoder
        self.label_decoder_cls = label_decoder
        self.intent_decoder_cls = intent_decoder

        self.batch_norm = nn.BatchNorm1d(self.z_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.embeds = common.ModuleList([embed_cls(
            vocab_size=vocab_size,
            dim=dim
        ) for embed_cls, vocab_size, dim in
            zip(self.embed_classes, self.vocab_sizes, self.word_dims)])

        self.encoder = self.word_encoder_cls(
            in_dim=self.word_dim,
            out_dim=self.z_dim
        )
        decoder_word = self.word_decoder_cls(
            in_dim=self.word_dim,
            hidden_dim=self.z_dim,
            out_dim=self.word_dim
        )
        decoder_label = self.label_decoder_cls(
            in_dim=self.label_dim + self.word_dim,
            hidden_dim=self.z_dim,
            out_dim=self.label_dim
        )
        decoder_intent = self.intent_decoder_cls(
            in_dim=self.z_dim + self.word_dim,
            out_dim=self.intent_dim
        )
        self.decoders = common.ModuleList([
            decoder_word,
            decoder_label,
            decoder_intent
        ])
        self.sampling = self.sampling_cls(
            in_dim=self.z_dim,
            out_dim=self.z_dim
        )
        self.sampling_s = None

    @staticmethod
    def apply_output_embed(embed, o):
        o_size = o.size()
        weight = embed.weight.t()
        o = torch.mm(o.view(-1, o_size[-1]), weight)
        return o.view(*o_size[:-1], -1)

    @staticmethod
    def randidx(x, size):
        """x is either integer or array-like probability distribution"""
        if isinstance(x, int):
            return torch.randint(0, x, size)
        else:
            x = np.random.choice(np.arange(len(x)), p=x, size=size)
            return torch.tensor(x)

    def forward_loss(self, w, l, i, lens):
        w = self.invoke(self.embeds[0], w)
        l = self.invoke(self.embeds[1], l)
        if self.should_dropout:
            w = self.dropout(w)
        h = self.invoke(self.encoder, w, lens)
        if self.should_batchnorm:
            h = self.batch_norm(h)
        if self.should_dropout:
            h = self.dropout(h)
        z = self.invoke(self.sampling, h)
        if self.word_dropout:
            probs = (self.word_dropout, 1 - self.word_dropout)
            rand_mask = self.randidx(probs, w.size()[:2]).to(w.device).float()
            rand_mask = rand_mask.unsqueeze(-1)
            input_word = rand_mask * w + (1 - rand_mask) * torch.zeros_like(w)
        else:
            input_word = w
        input_intent = torch.cat([common.expand_match(z, w, 1), w], 2)
        input_label = torch.cat([utils.roll_left(w), l], 2)
        os = (
            self.invoke(self.decoders[0], z, input_word, lens),
            self.invoke(self.decoders[1], z, input_label, lens),
            self.invoke(self.decoders[2], input_intent, lens)
        )
        logits = tuple(self.apply_output_embed(embed, o)
                       for embed, o in zip(self.embeds, os))
        yield "pass", logits

    def decode_single(self, dec, emb, z, bos, eos=None, cond=None, max_len=100):
        batch_size = z.size(0)
        x = z.new(batch_size, 1).long().fill_(bos)
        has_eos = x.new(batch_size).zero_().byte()
        lens = x.new(batch_size).fill_(x.size(1)).long()
        if cond is not None:
            max_len = min(cond.size(1), max_len)
        while has_eos.prod().item() != 1 and lens.max() < max_len:
            seq_len = x.size(1)
            x_emb = self.invoke(emb, x)
            if cond is not None:
                input = torch.cat([cond[:, :seq_len], x_emb], 2)
            else:
                input = x_emb
            o = self.invoke(dec, z, input, lens)
            o = o[:, -1].unsqueeze(1)
            logits = self.apply_output_embed(emb, o)
            logits = logits.squeeze(1)
            preds = logits.max(1)[1]
            x = torch.cat([x, preds.unsqueeze(1)], 1)
            if eos is not None:
                has_eos = (preds == eos) | has_eos
            lens += (1 - has_eos).long()
        return x, lens + 1

    def decode_single_beamsearch(self, dec, emb, z, bos, cond=None,
                                 eos=None, max_len=100, beam_size=3):
        max_float = 1e10
        min_float = -1e10
        softmax = nn.Softmax(2)
        batch_size = z.size(0)
        # forces the beam searcher to search from the first index only
        # in the beginning
        x = z.new(batch_size, 1, 1).long().fill_(bos)
        has_eos = x.new(batch_size, 1).zero_().byte()
        probs = z.new(batch_size, 1).fill_(1.0)
        lens = x.new(batch_size, 1).fill_(1).long()
        if cond is not None:
            max_len = min(cond.size(1), max_len)
        while has_eos.prod().item() != 1 and lens.max() < max_len:
            cur_beamsize, seq_len = x.size(1), x.size(2)
            x_emb = self.invoke(emb, x)
            if cond is not None:
                cond_exp = cond.unsqueeze(1).expand(-1, cur_beamsize, -1, -1)
                input = torch.cat([cond_exp[:, :, :seq_len], x_emb], 3)
            else:
                input = x_emb
            input = input.view(batch_size * cur_beamsize, seq_len, -1)
            z_exp = z.unsqueeze(1).expand(batch_size, cur_beamsize, -1) \
                .contiguous().view(batch_size * cur_beamsize, -1)
            o = self.invoke(dec, z_exp, input, lens.view(-1))[:, -1]
            x = x.view(batch_size, cur_beamsize, -1)
            o = o.view(batch_size, cur_beamsize, -1)
            logits = self.apply_output_embed(emb, o)
            # for beams that already generated <eos>, no more probability
            # depreciation.
            if eos is not None:
                eos_mask = has_eos.unsqueeze(-1).float()
                logits_eos = torch.full_like(logits, min_float)
                logits_eos[:, :, eos] = max_float
                logits = logits * (1 - eos_mask) + logits_eos * eos_mask
            # batch_size x beam_size x vocab_size
            p_vocab = probs.unsqueeze(-1) * softmax(logits)
            vocab_size = p_vocab.size(-1)
            # utilize 2d-flattened indices
            probs, idx = torch.sort(p_vocab.view(batch_size, -1), 1, True)
            probs, idx = probs[:, :beam_size], idx[:, :beam_size].long()
            beam_idx, preds = idx / vocab_size, idx % vocab_size
            x = torch.gather(x, 1,
                             beam_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            x = torch.cat([x, preds.unsqueeze(-1)], 2)
            if eos is not None:
                has_eos = torch.gather(has_eos, 1, beam_idx)
                has_eos = (preds == eos) | has_eos
            lens = torch.gather(lens, 1, beam_idx)
            lens += (1 - has_eos).long()
        return x, lens + 1, probs

    def predict_intent(self, z, w, lens):
        softmax = nn.Softmax(1)
        x = torch.cat([common.expand_match(z, w, 1), w], 2)
        o = self.invoke(self.decoders[2], x, lens)
        probs = softmax(self.apply_output_embed(self.embeds[2], o))
        pi, i = probs.max(1)
        return i, pi

    def encode(self, words, labels, intents, lens):
        w = self.invoke(self.embeds[0], words)
        h = self.invoke(self.encoder, w, lens)
        return self.invoke(self.sampling.mean, h), \
               self.invoke(self.sampling.std, h)

    def predict(self, w, lens, label_bos, beam_size=3):
        w = self.invoke(self.embeds[0], w)
        h = self.invoke(self.encoder, w, lens)
        z = self.sampling.mean(h)
        l, _, pl = self.decode_single_beamsearch(
            dec=self.decoders[1],
            emb=self.embeds[1],
            z=z,
            bos=label_bos,
            cond=utils.roll_left(w),
            beam_size=beam_size
        )
        l, pl = l[:, 0], pl[:, 0]
        i, pi = self.predict_intent(z, w, lens)
        return (l, i), (pl, pi)

    def generate(self, z, word_bos, label_bos, word_eos=None, max_len=100,
                 beam_size=3):
        batch_size = z.size(0)
        w, lens, pw = self.decode_single_beamsearch(
            dec=self.decoders[0],
            emb=self.embeds[0],
            z=z,
            bos=word_bos,
            eos=word_eos,
            max_len=max_len,
            beam_size=beam_size
        )
        w_flat = w.contiguous().view(batch_size * beam_size, -1)
        z_exp = z.unsqueeze(1).expand(batch_size, beam_size, -1).contiguous()
        z_exp = z_exp.view(batch_size * beam_size, -1)
        w_emb = self.invoke(self.embeds[0], w_flat)
        l, _, pl = self.decode_single_beamsearch(
            dec=self.decoders[1],
            emb=self.embeds[1],
            z=z_exp,
            bos=label_bos,
            max_len=w_flat.size(1),
            cond=utils.roll_left(w_emb),
            beam_size=beam_size
        )
        l = l[:, 0].view(batch_size, beam_size, -1)
        pl = pl[:, 0].view(batch_size, beam_size)
        i, pi = self.predict_intent(z_exp, w_emb, lens.view(-1))
        pi = pi.view(batch_size, beam_size)
        i = i.view(batch_size, beam_size)
        return (w, l, i, lens), (pw, pl, pi)

    def embeddings(self):
        for embed in self.embeds:
            yield embed