import os
import random
import logging
import argparse

import torch
import torch.nn as nn
import torch.utils.data as td

import utils
import model
import encode
import dataset
from model import embedding
from train import embeds
from . import neighbor


MODES = model.MODES

parser = argparse.ArgumentParser(
    fromfile_prefix_chars="@",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

group = parser.add_argument_group("Logging Options")
utils.add_logging_arguments(group, "generate")
group.add_argument("--argparse-filename",
                   type=str, default="generate-argparse.yml")
for mode in MODES:
    group.add_argument(f"--{mode}-filename", type=str, default=f"{mode}.txt")
    group.add_argument(f"--{mode}-probs-filename",
                       type=str, default=f"{mode}-probs.txt")
group.add_argument("--neighbors-filename", type=str, default="neighbors.txt")
group.add_argument("--show-progress", action="store_true", default=False)

group = parser.add_argument_group("Data Options")
for mode in MODES:
    group.add_argument(f"--{mode}-path", type=str, default=None)
    group.add_argument(f"--{mode}-vocab", type=str, required=True)
group.add_argument("--data-workers", type=int, default=8)
group.add_argument("--pin-memory", action="store_true", default=False)
group.add_argument("--seed", type=int, default=None)
group.add_argument("--unk", type=str, default="<unk>")
group.add_argument("--eos", type=str, default="<eos>")
group.add_argument("--bos", type=str, default="<bos>")

group = parser.add_argument_group("Generation Options")
group.add_argument("--validate", action="store_true", default=False)
group.add_argument("--ckpt-path", type=str, required=True)
group.add_argument("--batch-size", type=int, default=32)
group.add_argument("--save-dir", type=str, required=True)
group.add_argument("--beam-size", type=int, default=6)
group.add_argument("--beam-sample-topk", type=int, default=3)
group.add_argument("--samples", type=int, default=100)
group.add_argument("--nearest-neighbors", type=int, default=None)
group.add_argument("--nearest-neighbors-batch-size", type=int, default=16)
group.add_argument("--nn-workers", type=int, default=10)
group.add_argument("--max-length", type=int, default=30)
group.add_argument("--expand-vocab", action="store_true", default=False)
group.add_argument("--generation-type", default="gaussian",
                   choices=["gaussian", "posterior", "uniform"])
group.add_argument("--posterior-sampling-scale", type=float, default=1.0)
group.add_argument("--uniform-sampling-pa", type=float, default=1.0)
group.add_argument("--uniform-sampling-pm", type=float, default=1.0)

embeds.add_embed_arguments(group)

group.add_argument("--gpu", type=int, action="append", default=[])

group = parser.add_argument_group("Model Options")
model.add_arguments(group)


def prepare_dataset(args, vocab_sents, vocab_labels, vocab_intents):
    dset = dataset.TextSequenceDataset(
        paths=[args.word_path, args.label_path, args.intent_path],
        feats=["string", "tensor"],
        vocabs=[vocab_sents, vocab_labels, vocab_intents],
        pad_eos=args.eos,
        pad_bos=args.bos,
        unk=args.unk
    )
    return dset


def prepare_model(args, vocabs):
    mdl = model.create_model(args, vocabs)
    mdl.reset_parameters()
    ckpt = torch.load(args.ckpt_path)
    mdl.load_state_dict(ckpt)
    if args.expand_vocab:
        mdl_vocab = vocabs[0]
        mdl_emb = mdl.embeds[0].weight
        emb = embeds.get_embeddings(args)
        emb.preload()
        emb = {w: v for w, v in emb}
        for rword in [args.bos, args.eos, args.unk]:
            emb[rword] = mdl_emb[mdl_vocab.f2i.get(rword)].detach().numpy()
        vocab = utils.Vocabulary()
        utils.populate_vocab(emb.keys(), vocab)
        mdl.embeds[0] = embedding.BasicEmbedding(
            vocab_size=len(vocab),
            dim=mdl.word_dim,
            allow_padding=True
        )
        embeds._load_embeddings(mdl.embeds[0], vocab, emb.items())
    else:
        vocab = vocabs[0]
    return mdl, vocab


class Generator(object):

    def __init__(self, model, device, batch_size, beam_size, validate,
                 sent_vocab, label_vocab, intent_vocab, bos, eos, unk,
                 max_len, beam_topk):
        self.model = model
        self.device = device
        self.should_validate = validate
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.beam_topk = beam_topk
        self.sent_vocab = sent_vocab
        self.label_vocab = label_vocab
        self.intent_vocab = intent_vocab
        self.vocabs = [self.sent_vocab, self.label_vocab, self.intent_vocab]
        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.bos_idxs = [v.f2i.get(bos) for v in self.vocabs]
        self.eos_idxs = [v.f2i.get(eos) for v in self.vocabs]
        self.unk_idxs = [v.f2i.get(unk) for v in self.vocabs]
        self.max_len = max_len

    @property
    def module(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        else:
            return self.model

    def sample_z(self, num_samples):
        return torch.randn(num_samples, self.module.z_dim)

    def to_sent(self, idx, vocab):
        return " ".join(vocab.i2f.get(w, self.unk) for w in idx)

    def validate(self, data):
        """validate a single instance of sample"""
        if not self.should_validate:
            return True
        gen, probs = data
        sent, labels, intent = gen
        w_prob, l_prob, i_prob = probs
        words, labels = sent.split(), labels.split()
        def ensure_enclosed(x):
            return x[0] == self.bos and x[-1] == self.eos
        if not ensure_enclosed(words) or not ensure_enclosed(labels):
            return False
        if len(words) != len(labels):
            return False
        return True

    def _generate_step(self, z):
        gens, probs = self.model.generate(
            z=z,
            word_bos=self.bos_idxs[0],
            label_bos=self.bos_idxs[1],
            word_eos=self.eos_idxs[0],
            max_len=self.max_len,
            beam_size=self.beam_size
        )
        idx = random.randint(0, self.beam_topk - 1)
        gens, probs = [[x[:, idx] for x in g] for g in [gens, probs]]
        words, labels, intents, lens = gens
        gens = [(words, lens), (labels, lens),
                (intents.unsqueeze(-1), torch.ones_like(lens))]
        gens = [(x.cpu().tolist(), lens.cpu().tolist()) for x, lens in gens]
        gens = [[self.to_sent(gen[:l], v) for gen, l in zip(x, lens)]
                for (x, lens), v in zip(gens, self.vocabs)]
        gens = list(zip(*gens))
        probs = list(zip(*[p.tolist() for p in probs]))
        return gens, probs

    def generate(self, num_samples, sampler=None):
        """
        Generates `num_samples` samples. If sampler is specified, z returned by
        the sampler is used to decode. This function guarantees at least
        `num_samples` number of samples are generated after culling invalid
        sequences.
        :param num_samples: int
        :param sampler: int -> [batch_size, z_dim] torch.FloatTensor
        :return:
        """
        if sampler is None:
            sampler = self.sample_z
        self.model.train(False)
        samples = []
        progress = utils.tqdm(total=num_samples, desc="generating")
        while len(samples) < num_samples:
            batch_size = min(num_samples - len(samples), self.batch_size)
            z = sampler(batch_size)
            z = z.to(self.device)
            gens = list(zip(*self._generate_step(z)))
            unfiltered_size = len(gens)
            gens = list(filter(self.validate, gens))
            if len(gens) < unfiltered_size:
                logging.info(f"filtered out {unfiltered_size - len(gens)} "
                             "instances...")
            progress.update(len(gens))
            samples.extend(gens)
        progress.close()
        gens, probs = list(zip(*samples))
        return list(zip(*gens)), list(zip(*probs))

    def generate_from(self, zs):
        self.model.train(False)
        res = []
        progress = utils.tqdm(total=len(zs), desc="generating")
        for i in range(0, len(zs), self.batch_size):
            z = zs[i:i + self.batch_size]
            progress.update(len(z))
            z = z.to(self.device)
            gens, probs = self._generate_step(z)
            res.extend(list(zip(gens, probs)))
        progress.close()
        gens, probs = list(zip(*res))
        return list(zip(*gens)), list(zip(*probs))


def save(args, gens, probs, neighbors=None):
    sents, labels, intents = gens
    sents = [" ".join(sent.split()[1:-1]) for sent in sents]
    labels = [" ".join(label.split()[1:-1]) for label in labels]
    probs = [list(map(str, p)) for p in probs]
    if neighbors is not None:
        neighbors = [[" ".join(n.split()[1:-1]) for n in nn]
                     for nn in neighbors]
    samples = [sents, labels, intents] + probs
    fnames = [args.word_filename, args.label_filename,
              args.intent_filename, args.word_probs_filename,
              args.label_probs_filename, args.intent_probs_filename]
    paths = [os.path.join(args.save_dir, fn) for fn in fnames]
    for data, path in zip(samples, paths):
        with open(path, "w") as f:
            for sample in data:
                f.write(f"{sample}\n")
    if neighbors is not None:
        neighbors_path = os.path.join(args.save_dir, args.neighbors_filename)
        with open(neighbors_path, "w") as f:
            for neighbor in neighbors:
                delim = "\t"
                f.write(f"{delim.join(neighbor)}\n")


def report_stats(args, sents, neighbors):
    exists = [sent in nn for sent, nn in zip(sents, neighbors)]
    exists_ratio = sum(1.0 if e else 0.0 for e in exists) / len(exists)
    logging.info(f"Out of {len(exists)} samples, "
                 f"{exists_ratio * 100:.2f}% of them exist in the original data")


class MultivariateGaussianMixtureSampler(object):

    def __init__(self, means: torch.Tensor, stds: torch.Tensor, scale=1.0):
        assert len(means) == len(stds)
        self.means = means
        self.stds = stds
        self.scale = scale

    def __len__(self):
        return len(self.means)

    def __call__(self, num_samples):
        randidx = torch.randint(0, len(self), (num_samples, )).long()
        means = torch.index_select(self.means, 0, randidx)
        stds = torch.index_select(self.stds, 0, randidx)
        return torch.randn_like(stds) * stds * self.scale + means


class UniformNoiseSampler(object):

    """Kurata et al. 2016"""
    def __init__(self, embs, pa=1.0, pm=1.0):
        self.embs = embs
        self.pa = pa
        self.pm = pm

    def __len__(self):
        return len(self.embs)

    def __call__(self, num_samples):
        randidx = torch.randint(0, len(self), (num_samples, )).long()
        embs = torch.index_select(self.embs, 0, randidx)
        a = (torch.rand_like(embs) * 2 - 1) * self.pa
        m = (torch.rand_like(embs) * 2 - 1) * self.pm + 1
        return embs * m + a


def create_dataloader(args, vocabs):
    dset = prepare_dataset(args, *vocabs)
    return td.DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.data_workers,
        collate_fn=dataset.TextSequenceBatchCollator(
            pad_idxs=[len(v) for v in vocabs]
        ),
        pin_memory=args.pin_memory
    )

def generate(args):
    devices = utils.get_devices(args.gpu)
    if args.seed is not None:
        utils.manual_seed(args.seed)

    logging.info("Loading data...")
    vocab_paths = [args.word_vocab, args.label_vocab, args.intent_vocab]
    vocabs = [utils.load_pkl(v) for v in vocab_paths]
    dataloader = None

    logging.info("Initializing generation environment...")
    model, vocabs[0] = prepare_model(args, vocabs)
    model = utils.to_device(model, devices)
    encoder = encode.Encoder(
        model=model,
        device=devices[0],
        batch_size=args.batch_size
    )
    generator = Generator(
        model=model,
        device=devices[0],
        batch_size=args.batch_size,
        sent_vocab=vocabs[0],
        label_vocab=vocabs[1],
        intent_vocab=vocabs[2],
        bos=args.bos,
        eos=args.eos,
        unk=args.unk,
        max_len=args.max_length,
        beam_size=args.beam_size,
        beam_topk=args.beam_sample_topk,
        validate=args.validate
    )

    logging.info("Commencing generation...")
    if args.generation_type in {"posterior", "uniform"}:
        if dataloader is None:
            dataloader = create_dataloader(args, vocabs)
    sampler = utils.map_val(args.generation_type, {
        "gaussian": lambda: None,
        "posterior": lambda: MultivariateGaussianMixtureSampler(
            *encoder.encode(dataloader),
            scale=args.posterior_sampling_scale
        ),
        "uniform": lambda: UniformNoiseSampler(
            encoder.encode(dataloader)[0],
            pa=args.uniform_sampling_pa,
            pm=args.uniform_sampling_pm
        )
    }, name="sampler")()
    with torch.no_grad():
        gens, probs = generator.generate(args.samples, sampler)
    if args.nearest_neighbors is not None:
        if dataloader is None:
            dataloader = create_dataloader(args, vocabs)
        sents = [data["string"][0] for data in dataloader.dataset]
        searcher = neighbor.PyTorchPCASearcher(
            pca_dim=100,
            sents=sents,
            num_neighbors=args.nearest_neighbors,
            batch_size=args.nearest_neighbors_batch_size,
            device=devices[0]
        )
        neighbors = searcher.search(gens[0])
    else:
        neighbors = None
    report_stats(args, gens[0], neighbors)
    save(args, gens, probs, neighbors)

    logging.info("Done!")


if __name__ == "__main__":
    generate(utils.initialize_script(parser))