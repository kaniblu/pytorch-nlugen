from . import common
from . import fusion
from . import gaussian
from . import discrete_encoder
from . import discrete_decoder


class AbstractMultimodalVariationalAutoencoder(common.Module):

    """
    xs: tuple of [batch_size, seq_len] LongTensor
    lens: tuple of [batch_size] LongTensor
    """
    def __init__(self, num_modes, z_dim, vocab_sizes, word_dims):
        super(AbstractMultimodalVariationalAutoencoder, self).__init__()
        assert len(vocab_sizes) == len(word_dims) == num_modes
        self.num_modes = num_modes
        self.z_dim = z_dim
        self.vocab_sizes = vocab_sizes
        self.word_dims = word_dims

    def forward_loss(self, xs, lens):
        raise NotImplementedError()


class MultimodalVariationalAutoencoder(AbstractMultimodalVariationalAutoencoder):

    name = "multimodal-variational-autoencoder"

    def __init__(self, *args, singulars=True,
                 sampling=gaussian.AbstractGaussianSampling,
                 encoders=(discrete_encoder.AbstractDiscreteSequenceEncoder, ),
                 decoders=(discrete_decoder.AbstractDiscreteSequenceDecoder, ),
                 fusion=fusion.BaseFusion, **kwargs):
        super(MultimodalVariationalAutoencoder, self).__init__(*args, **kwargs)
        assert len(encoders) == len(decoders) == self.num_modes
        self.singulars = singulars
        self.encoders_cls = encoders
        self.decoders_cls = decoders
        self.fusion_cls = fusion
        self.sampling_cls = sampling
        self.encoders = common.ModuleList([
            cls(
                vocab_size=vocab_size,
                word_dim=word_dim,
                hidden_dim=self.z_dim
            ) for cls, vocab_size, word_dim in \
            zip(self.encoders_cls, self.vocab_sizes, self.word_dims)
        ])
        self.decoders = common.ModuleList([
            cls(
                vocab_size=vocab_size,
                word_dim=word_dim,
                hidden_dim=self.z_dim
            ) for cls, vocab_size, word_dim in \
            zip(self.decoders_cls, self.vocab_sizes, self.word_dims)
        ])
        self.sampling = self.sampling_cls(
            in_dim=self.z_dim,
            out_dim=self.z_dim
        )
        self.sampling_singulars = None
        if self.singulars:
            self.sampling_singulars = common.ModuleList([self.sampling_cls(
                in_dim=self.z_dim,
                out_dim=self.z_dim
            ) for _ in range(self.num_modes)])
        self.fusion = self.fusion_cls(
            in_dims=(self.z_dim, ) * self.num_modes,
            out_dim=self.z_dim
        )

    def forward_loss(self, xs, lens):
        hs = [self.invoke(enc, x, ls)
              for enc, x, ls in zip(self.encoders, xs, lens)]
        h_fuse = self.invoke(self.fusion, *hs)
        z = self.invoke(self.sampling, h_fuse)
        xs = [x[:, :-1].contiguous() for x in xs]
        lens = [ls - 1 for ls in lens]
        logits_fuse = [self.invoke(dec, z, x, ls)
                       for dec, x, ls in zip(self.decoders, xs, lens)]
        if self.singulars:
            zs = [self.invoke(sampling, h)
                  for sampling, h in zip(self.sampling_singulars, hs)]
            logits_singular = [self.invoke(dec, z, x, ls)
                               for dec, z, x, ls in
                               zip(self.decoders, zs, xs, lens)]
            yield "pass", (logits_fuse, logits_singular)
        else:
            yield "pass", logits_fuse

    def decode(self, z, bos_idxs, eos_idxs, max_len):
        return [dec.decode(z, bos, eos, max_len)
                for dec, bos, eos in zip(self.decoders, bos_idxs, eos_idxs)]