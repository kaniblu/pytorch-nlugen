import os
import logging
import argparse
import itertools
import importlib
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as op
import torch.utils.data as td

import utils
import model
import encode
import dataset
import predict
import evaluate
import generate.__main__
import generate.analyze
from . import embeds
MODES = model.MODES

parser = argparse.ArgumentParser(
    fromfile_prefix_chars="@",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

group = parser.add_argument_group("Logging Options")
utils.add_logging_arguments(group, "train-luvae")
group.add_argument("--show-progress", action="store_true", default=False)

group = parser.add_argument_group("Model Options")
model.add_arguments(group)

group = parser.add_argument_group("Data Options")
for mode in MODES:
    group.add_argument(f"--{mode}-path", type=str, required=True)
    group.add_argument(f"--{mode}-vocab", type=str, default=None)
group.add_argument("--vocab-limit", type=int, default=None)
group.add_argument("--data-workers", type=int, default=8)
group.add_argument("--pin-memory", action="store_true", default=False)
group.add_argument("--shuffle", action="store_true", default=False)
group.add_argument("--seed", type=int, default=None)
group.add_argument("--unk", type=str, default="<unk>")
group.add_argument("--eos", type=str, default="<eos>")
group.add_argument("--bos", type=str, default="<bos>")

group = parser.add_argument_group("Training Options")
group.add_argument("--save-dir", type=str, required=True)
group.add_argument("--save-period", type=int, default=1)
group.add_argument("--batch-size", type=int, default=32)
group.add_argument("--epochs", type=int, default=12)
group.add_argument("--kld-annealing", type=float, default=None)
group.add_argument("--optimizer", type=str, default="adam",
                   choices=["adam", "adamax", "adagrad", "adadelta"])
group.add_argument("--learning-rate", type=float, default=None)
group.add_argument("--tensorboard", action="store_true", default=False)
group.add_argument("--gpu", type=int, action="append", default=[])

group = parser.add_argument_group("Validation Options")
group.add_argument("--validate", action="store_true", default=False)
group.add_argument("--val-period", default=1, type=int)
for mode in MODES:
    group.add_argument(f"--val-{mode}-path", type=str)
group.add_argument("--validate-understanding", action="store_true", default=False)
group.add_argument("--validate-generation", action="store_true", default=False)
group.add_argument("--validate-random-sampling", action="store_true", default=False)
group.add_argument("--validate-fixed-sampling", action="store_true", default=False)
group.add_argument("--validate-autoencoding-sampling", action="store_true", default=False)
group.add_argument("--validate-posterior-sampling", action="store_true", default=False)
group.add_argument("--plot-embedding", action="store_true", default=False)
group.add_argument("--beam-size", type=int, default=1)
group.add_argument("--max-length", type=int, default=50)
group.add_argument("--num-generations", type=int, default=100)
group.add_argument("--num-random-samples", type=int, default=10)
group.add_argument("--num-fixed-samples", type=int, default=10)
group.add_argument("--num-autoencoding-samples", type=int, default=10)
group.add_argument("--num-posterior-samples", type=int, default=10)
group.add_argument("--generation-analysis-encoder", default="use",
                   choices=["use", "wordembed"])

group = parser.add_argument_group("Word Embeddings Options")
for mode in MODES:
    group.add_argument(f"--{mode}-pretrained", action="store_true", default=False)
    embeds.add_embed_arguments(group, mode)


def create_dataloader(args, vocabs=None, val=False):
    if not val:
        prefix = ""
        shuffle = args.shuffle
    else:
        prefix = "val_"
        shuffle = False
    paths = [getattr(args, f"{prefix}{mode}_path") for mode in MODES]
    if vocabs is None:
        vocabs = [getattr(args, f"{prefix}{mode}_vocab") for mode in MODES]
        vocabs = [utils.load_pkl(v) if v is not None else None for v in vocabs]
    dset = dataset.TextSequenceDataset(
        paths=paths,
        feats=["string", "tensor"],
        vocabs=vocabs,
        vocab_limit=args.vocab_limit,
        pad_eos=args.eos,
        pad_bos=args.bos,
        unk=args.unk,
    )
    if vocabs is None:
        vocabs = dset.vocabs
    collator = dataset.TextSequenceBatchCollator(
        pad_idxs=[len(v) for v in vocabs]
    )
    return td.DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.data_workers,
        collate_fn=collator,
        pin_memory=args.pin_memory
    )


def prepare_model(args, vocabs, wem):
    mdl = model.create_model(args, vocabs)
    mdl.reset_parameters()
    for mode, vocab, emb in zip(MODES, vocabs, mdl.embeddings()):
        if not getattr(args, f"{mode}_pretrained"):
            continue
        we = embeds.get_embeddings(utils.filter_namespace_prefix(args, mode))
        we = wem[we]
        embeds._load_embeddings(emb, vocab, we)
    return mdl


def get_optimizer_cls(args):
    kwargs = dict()
    if args.learning_rate is not None:
        kwargs["lr"] = args.learning_rate
    return utils.map_val(args.optimizer, {
        "adam": lambda p: op.Adam(p, **kwargs),
        "adamax": lambda p: op.Adamax(p, **kwargs),
        "adagrad": lambda p: op.Adagrad(p, **kwargs),
        "adadelta": lambda p: op.Adadelta(p, **kwargs)
    }, "optimizer")


def normalize(x):
    return x / sum(x)


def randidx(x, size):
    """x is either integer or array-like probability distribution"""
    if isinstance(x, int):
        return torch.randint(0, x, size)
    else:
        return np.random.choice(np.arange(len(x)), p=x, size=size)


def xor(a, b):
    return (a and not b) or (b and not a)


def has_none(l):
    return any(x is None for x in l)


class Trainer(object):

    def __init__(self, debug, model, device, vocabs, epochs, save_dir,
                 save_period, optimizer_cls=op.Adam, enable_tensorboard=False,
                 show_progress=True, kld_annealing=None, tensor_key="tensor",
                 word_droprate=0.5,
                 encoder: encode.Encoder=None,
                 generator: generate.__main__.Generator=None,
                 gen_analyzer: generate.analyze.Analyzer=None,
                 predictor: predict.Predictor=None,
                 num_generations=None,
                 num_random_samples=None, num_fixed_samples=None,
                 num_posterior_samples=None,
                 num_ae_samples=None, plot_embedding=False, val_period=1):
        self.word_droprate = word_droprate
        self.debug = debug
        self.model = model
        self.device = device
        self.epochs = epochs
        self.vocabs = vocabs
        self.save_dir = save_dir
        self.save_period = save_period
        self.optimizer_cls = optimizer_cls
        self.tensor_key = tensor_key
        self.kld_annealing = kld_annealing
        self.enable_tensorboard = enable_tensorboard
        self.cross_entropies = [nn.CrossEntropyLoss(ignore_index=len(vocab))
                                for vocab in vocabs]
        self.show_progress = show_progress
        self.unk = "<unk>"
        self.encoder = encoder
        self.predictor = predictor
        self.generator = generator
        self.gen_z = None
        self.num_generations = num_generations
        self.num_samples = (
            num_random_samples,
            num_fixed_samples,
            num_ae_samples,
            num_posterior_samples
        )
        self.val_period = val_period
        self.should_validate = self.predictor is not None or \
            self.generator is not None
        self.gen_analyzer = gen_analyzer
        self.plot_embedding = plot_embedding
        self.plots = []

        assert not xor(self.gen_analyzer is None, self.generator is None), \
            "analyzer must be specified when generator is given"

        assert num_ae_samples and encoder is not None or not num_ae_samples

        if self.num_generations is None:
            self.num_generations = 0
        self.num_samples = tuple(0 if x is None else x
                                 for x in self.num_samples)
        if self.generator is not None and self.num_samples[1]:
            self.gen_z = torch.randn(self.num_samples[1], self.module.z_dim)

        if self.enable_tensorboard:
            self.tensorboard = importlib.import_module("tensorboardX")
            self.writer = self.tensorboard.SummaryWriter(log_dir=self.save_dir)

    @property
    def module(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        else:
            return self.model

    def trainable_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                yield p

    def prepare_batch(self, batch):
        data = batch[self.tensor_key]
        data = [(x.to(self.device), lens.to(self.device)) for x, lens in data]
        (w, w_lens), (l, l_lens), (i, i_lens) = data
        batch_size = w.size(0)
        if self.debug:
            assert (w_lens == l_lens).sum().item() == batch_size
            assert (i_lens == 3).sum().item() == batch_size
        return batch_size, (w, l, i[:, 1], w_lens)

    def snapshot(self, eidx):
        state_dict = self.module.state_dict()
        path = os.path.join(self.save_dir, f"checkpoint-e{eidx:02d}")
        torch.save(state_dict, path)
        logging.info(f"checkpoint saved to '{path}'.")

    def get_header(self, eidx=None):
        if eidx is not None:
            header = f"[s{self.global_step}/e{eidx}]"
        else:
            header = f"[s{self.global_step}]"
        return header

    def report_stats(self, stats, eidx=None):
        if self.enable_tensorboard:
            for k, v in stats.items():
                self.writer.add_scalar(k.replace("-", "/"), v, self.global_step)
        stats_str = {k: f"{v:.4f}" for k, v in stats.items()}
        desc = utils.join_dict(stats_str, ", ", "=")
        header = self.get_header(eidx)
        logging.info(f"{header} {desc}")
        return desc

    def report_samples(self, words, labels, intents, name=None, eidx=None):
        text = "\n".join(f"({i:03d}/{intent}) {word}"
                         for i, (word, label, intent) in
                         enumerate(zip(words, labels, intents)))
        header = self.get_header(eidx)
        logging.info(f"{header} {name}:\n{text}")
        if self.enable_tensorboard:
            self.writer.add_text(name, text.replace("\n", "\n\n"),
                                 global_step=self.global_step)

    def calculate_celoss(self, ce, logits, targets):
        logit_size = logits.size(-1)
        logits = logits.view(-1, logit_size)
        targets = targets.view(-1)
        return ce(logits, targets)

    def calculate_losses(self, logits_lst, gold_lst):
        losses = {
            f"loss-{mode}": self.calculate_celoss(ce, logits, gold)
            for mode, ce, logits, gold in
            zip(MODES, self.cross_entropies, logits_lst, gold_lst)
        }
        return losses

    def prepare_gold(self, w, l, i, lens):
        batch_size, seq_len = w.size()
        w_ignore = w.new(batch_size, 1).fill_(len(self.vocabs[0]))
        w = torch.cat([w[:, 1:], w_ignore], 1)
        l_ignore = l.new(batch_size, 1).fill_(len(self.vocabs[1]))
        l = torch.cat([l[:, 1:], l_ignore], 1)
        return w, l, i, lens

    def report_embeddings(self, embs, w, l, i,
                          name="default", final=False, goldw=None):
        assert len(embs) == len(w) == len(l) == len(i)
        if goldw is not None:
            assert len(embs) == len(goldw)
        num_items = len(embs)
        names = [name] * num_items
        if goldw is None:
            goldw = [""] * num_items
        self.plots.append((embs, w, l, i, names, goldw))

        if final:
            embs_lst, w_lst, l_lst, i_lst, names_lst, goldw_lst = zip(*self.plots)
            embs = torch.cat(embs_lst, 0)
            w = list(itertools.chain(*w_lst))
            l = list(itertools.chain(*l_lst))
            i = list(itertools.chain(*i_lst))
            names = list(itertools.chain(*names_lst))
            goldw = list(itertools.chain(*goldw_lst))
            self.writer.add_embedding(
                mat=embs,
                metadata=list(zip(w, l, i, names, goldw)),
                metadata_header=(
                    "word",
                    "label",
                    "intent",
                    "name",
                    "word-gold"
                ),
                global_step=self.global_step
            )
            self.plots = []

    def validate_generation(self, eidx):
        num_samples = max(self.num_samples[0], self.num_generations)
        if not num_samples:
            return
        zs = torch.randn(num_samples, self.module.z_dim)
        (w, l, i), _ = self.generator.generate_from(zs)
        ws = list(map(self.strip_beos, w))
        ls = list(map(self.strip_beos, l))
        if self.plot_embedding:
            self.report_embeddings(zs, ws, ls, i, name="random",
                                   goldw=ws)
        if self.num_generations:
            genstats = self.gen_analyzer.analyze(ws)
            genstats = {f"val-gen-{k}": v for k, v in genstats.items()}
            self.report_stats(genstats, eidx)
        if self.num_samples[0]:
            samples = list(zip(w, l, i))[:self.num_samples[0]]
            w, l, i = list(zip(*samples))
            self.report_samples(w, l, i, name="val-rand", eidx=eidx)

    def validate_fixed_samples(self, eidx):
        (w, l, i), probs = self.generator.generate_from(self.gen_z)
        ws = list(map(self.strip_beos, w))
        ls = list(map(self.strip_beos, l))
        if self.plot_embedding:
            self.report_embeddings(self.gen_z, ws, ls, i, name="fixed")
        if self.num_samples[1]:
            samples = list(zip(w, l, i))[:self.num_samples[1]]
            w, l, i = list(zip(*samples))
            self.report_samples(w, l, i, name="val-fixed", eidx=eidx)

    def validate_autoencoding_samples(self, dataloader, eidx):
        idx = np.random.permutation(np.arange(len(dataloader.dataset)))
        idx = idx[:self.num_samples[2]]
        ds = [dataloader.dataset[i] for i in idx]
        goldw, goldl, goldi = zip(*[data["string"] for data in ds])
        goldi = list(map(self.strip_beos, goldi))
        dataloader = td.DataLoader(
            dataset=ds,
            batch_size=dataloader.batch_size,
            collate_fn=dataset.TextSequenceBatchCollator(
                pad_idxs=[len(v) for v in self.vocabs]
            ),
            num_workers=dataloader.num_workers
        )
        zs = self.encoder.encode(dataloader)[0]
        (w, l, i), probs = self.generator.generate_from(zs)
        ws = list(map(self.strip_beos, w))
        ls = list(map(self.strip_beos, l))
        if self.plot_embedding:
            self.report_embeddings(zs, ws, ls, i,
                final=False,
                name="dataset",
                goldw=list(map(self.strip_beos, goldw))
            )
        if self.num_samples[2]:
            self.report_samples(w, l, i, name="val-ae-pred", eidx=eidx)
            self.report_samples(goldw, goldl, goldi,
                                name="val-ae-gold", eidx=eidx)

    def validate_posterior_samples(self, dataloader, eidx):
        idx = np.random.permutation(np.arange(len(dataloader.dataset)))
        idx = idx[:self.num_samples[3]]
        ds = [dataloader.dataset[i] for i in idx]
        goldw, goldl, goldi = zip(*[data["string"] for data in ds])
        goldi = list(map(self.strip_beos, goldi))
        dataloader = td.DataLoader(
            dataset=ds,
            batch_size=dataloader.batch_size,
            collate_fn=dataset.TextSequenceBatchCollator(
                pad_idxs=[len(v) for v in self.vocabs]
            ),
            num_workers=dataloader.num_workers
        )
        zs_mean, zs_std = self.encoder.encode(dataloader)
        zs = torch.randn_like(zs_std) * zs_std + zs_mean
        (w_mean, l_mean, i_mean), probs_mean = \
            self.generator.generate_from(zs_mean)
        (w, l, i), probs = self.generator.generate_from(zs)
        ws_mean = list(map(self.strip_beos, w_mean))
        ls_mean = list(map(self.strip_beos, l_mean))
        ws = list(map(self.strip_beos, w))
        ls = list(map(self.strip_beos, l))
        if self.plot_embedding:
            self.report_embeddings(zs, ws, ls, i,
                final=False,
                name="posterior",
                goldw=list(map(self.strip_beos, goldw))
            )
            self.report_embeddings(zs_mean, ws_mean, ls_mean, i_mean,
                final=True,
                name="posterior-mean",
                goldw=list(map(self.strip_beos, goldw))
            )
        if self.num_samples[3]:
            self.report_samples(w, l, i, name="val-posterior-pred", eidx=eidx)
            self.report_samples(w_mean, l_mean, i_mean,
                                name="val-posterior-pred-mean", eidx=eidx)
            self.report_samples(goldw, goldl, goldi,
                                name="val-posterior-gold", eidx=eidx)

    @staticmethod
    def strip_beos(sent):
        return " ".join(sent.split()[1:-1])

    def validate(self, eidx, dataloader=None):
        with torch.no_grad():
            if self.generator is not None:
                self.validate_generation(eidx)
                if self.num_samples[1]:
                    self.validate_fixed_samples(eidx)
                if self.num_samples[2]:
                    self.validate_autoencoding_samples(dataloader, eidx)
                if self.num_samples[3]:
                    self.validate_posterior_samples(dataloader, eidx)
            if self.predictor is not None:
                (predl, predi), (pl, pi) = self.predictor.predict(dataloader)
                goldl, goldi = zip(*[data["string"][1:]
                                     for data in dataloader.dataset])
                predl, goldl, goldi = [list(map(self.strip_beos, x))
                                       for x in [predl, goldl, goldi]]
                lustats = evaluate.evaluate(goldl, goldi, predl, predi)
                lustats = {
                    "val-lu-intent": lustats["intent-classification"]["overall"],
                    "val-lu-slot": lustats["slot-labeling"]["overall"],
                    "val-lu-sent": {"acc": lustats["sentence-understanding"]}
                }
                lustats = {
                    f"{k}-{k2}": v2
                    for k, v in lustats.items()
                    for k2, v2 in v.items()
                }
                self.report_stats(lustats, eidx)

    def train(self, dataloader, val_dataloader=None):
        self.global_step = 0
        optimizer = self.optimizer_cls(list(self.trainable_params()))
        progress_global = utils.tqdm(
            total=self.epochs,
            desc=f"training {self.epochs} epochs",
            disable=not self.show_progress
        )
        if self.kld_annealing is not None:
            kld_scale = 0.0
        else:
            kld_scale = 1.0

        for eidx in range(1, self.epochs + 1):
            self.module.train(True)
            self.local_step = 0
            stats_cum = collections.defaultdict(float)
            progress_global.update(1)
            progress_local = utils.tqdm(
                total=len(dataloader.dataset),
                desc=f"training an epoch",
                disable=not self.show_progress
            )
            for batch in dataloader:
                optimizer.zero_grad()
                batch_size, (w, l, i, lens) = self.prepare_batch(batch)
                self.global_step += batch_size
                self.local_step += batch_size
                progress_local.update(batch_size)
                ret = self.model(w, l, i, lens)
                batch_gold = self.prepare_gold(w, l, i, lens)
                logits, loss_kld = ret.get("pass"), ret.get("loss")
                losses = self.calculate_losses(
                    logits_lst=logits,
                    gold_lst=batch_gold
                )
                loss = sum(losses.values())
                if loss_kld is not None:
                    loss += kld_scale * loss_kld.mean()
                loss.backward()
                optimizer.step()

                stats = losses
                if loss_kld is not None:
                    stats["loss-kld"] = kld_scale * loss_kld.mean().item()
                    stats["weight-kld"] = kld_scale
                for k, v in stats.items():
                    stats_cum[k] += v * batch_size
                self.report_stats(stats)

            progress_local.close()
            stats_cum = {f"{k}-epoch": v / self.local_step for k, v in stats_cum.items()}
            self.report_stats(stats_cum, eidx)

            if self.kld_annealing is not None:
                kld_scale += self.kld_annealing
                kld_scale = min(1.0, kld_scale)

            if eidx % self.save_period == 0:
                self.snapshot(eidx)

            if self.should_validate and eidx % self.val_period == 0:
                if self.debug and self.predictor is not None:
                    assert val_dataloader is not None, \
                        "must provide dataloader when validation is on"
                self.validate(eidx, val_dataloader)

        progress_global.close()


def report_model(trainer):
    params = sum(np.prod(p.size()) for p in trainer.trainable_params())
    logging.info(f"Number of parameters: {params:,}")
    logging.info(f"{trainer.module}")


def train(args):
    devices = utils.get_devices(args.gpu)
    if args.seed is not None:
        utils.manual_seed(args.seed)

    logging.info("Loading data...")
    dataloader = create_dataloader(args)
    vocabs = dataloader.dataset.vocabs
    val_dataloader = None
    if args.validate and (args.validate_understanding
            or args.validate_posterior_sampling
            or args.validate_autoencoding_sampling):
        val_dataloader = create_dataloader(args, vocabs, True)
    fnames = [f"{mode}.vocab" for mode in MODES]
    for vocab, fname in zip(vocabs, fnames):
        utils.save_pkl(vocab, os.path.join(args.save_dir, fname))

    logging.info("Initializing training environment...")
    wem = embeds.WordEmbeddingManager()
    mdl = prepare_model(args, vocabs, wem)
    optimizer_cls = get_optimizer_cls(args)
    generator, analyzer, encoder, predictor = None, None, None, None
    if args.validate:
        if args.validate_generation or args.validate_fixed_sampling or \
                args.validate_random_sampling or \
                args.validate_autoencoding_sampling or \
                args.validate_posterior_sampling:
            generator = generate.__main__.Generator(
                model=mdl,
                device=devices[0],
                batch_size=args.batch_size,
                beam_size=args.beam_size,
                validate=False,
                sent_vocab=vocabs[0],
                label_vocab=vocabs[1],
                intent_vocab=vocabs[2],
                bos=args.bos,
                eos=args.eos,
                unk=args.unk,
                max_len=args.max_length,
                beam_topk=1
            )
            if args.generation_analysis_encoder == "use":
                use_encoder = generate.analyze.UniversalSentenceEncoder(
                    batch_size=args.batch_size,
                    minimal_gpumem=True
                )
                analyzer = generate.analyze.Analyzer(
                    encoder=use_encoder,
                    device=devices[0],
                )
            elif args.generation_analysis_encoder == "wordembed":
                we = embeds.get_embeddings(utils.filter_namespace_prefix(args, "word"))
                emb_encoder = generate.analyze.WordEmbeddingSentenceEncoder(
                    word_embs=wem[we]
                )
                analyzer = generate.analyze.Analyzer(
                    encoder=emb_encoder,
                    device=devices[0]
                )
            else:
                raise ValueError(f"unrecognized analysis encoder type: "
                                 f"{args.generation_analysis_encoder}")
        if args.validate_autoencoding_sampling or \
                args.validate_posterior_sampling:
            encoder = encode.Encoder(
                model=mdl,
                device=devices[0],
                batch_size=args.batch_size
            )
        if args.validate_understanding:
            predictor = predict.Predictor(
                model=mdl,
                device=devices[0],
                batch_size=args.batch_size,
                beam_size=args.beam_size,
                sent_vocab=vocabs[0],
                label_vocab=vocabs[1],
                intent_vocab=vocabs[2],
                bos=args.bos,
                eos=args.eos,
                unk=args.unk,
            )
    trainer = Trainer(
        debug=True,
        model=utils.to_device(mdl, devices),
        device=devices[0],
        vocabs=vocabs,
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_period=args.save_period,
        optimizer_cls=optimizer_cls,
        tensor_key="tensor",
        enable_tensorboard=args.tensorboard,
        show_progress=args.show_progress,
        kld_annealing=args.kld_annealing,
        generator=generator,
        gen_analyzer=analyzer,
        predictor=predictor,
        encoder=encoder,
        num_generations=args.num_generations
        if args.validate_generation else None,
        num_random_samples=args.num_random_samples
        if args.validate_random_sampling else None,
        num_ae_samples=args.num_autoencoding_samples
        if args.validate_autoencoding_sampling else None,
        num_fixed_samples=args.num_fixed_samples
        if args.validate_fixed_sampling else None,
        num_posterior_samples=args.num_posterior_samples
        if args.validate_posterior_sampling else None,
        val_period=args.val_period,
        plot_embedding=args.plot_embedding
    )
    report_model(trainer)

    logging.info("Commencing training luvae...")
    trainer.train(dataloader, val_dataloader)

    logging.info("Done!")


if __name__ == "__main__":
    train(utils.initialize_script(parser))