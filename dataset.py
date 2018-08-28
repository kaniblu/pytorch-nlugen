import io
import gzip
import collections

import torch
import torch.utils.data as td

import utils


class TextFileReader(utils.UniversalFileReader):
    def __init__(self):
        super(TextFileReader, self).__init__("txt")

    def open_txt(self, path):
        return open(path, "r")

    def open_gz(self, path):
        return io.TextIOWrapper(gzip.open(path, "r"))


def pad_tensor(x, size, pad_idx=0):
    if size <= 0:
        return x

    padding = x.new(size).fill_(pad_idx)
    return torch.cat([x, padding])


def pad_sequences(x, max_len=None, pad_idx=0):
    if max_len is None:
        max_len = max(map(len, x))
    x = [pad_tensor(t, max_len - len(t), pad_idx) for t in x]
    x = torch.stack(x)

    return x


class TextSequenceDataset(td.Dataset):
    FEATURES = {
        "string",
        "tensor"
    }

    def __init__(self, paths, feats=None, vocabs=None, vocab_limit=None,
                 pad_bos=None, pad_eos=None, unk="<unk>"):
        if feats is None:
            feats = self.FEATURES
        if not isinstance(vocabs, collections.Sequence):
            vocabs = [vocabs] * len(paths)
        self.paths = paths
        self.feats = feats
        self.vocabs = vocabs
        self.vocab_limit = vocab_limit
        self.pad_eos = pad_eos
        self.pad_bos = pad_bos
        self.unk = unk
        self.unk_idxs = [None] * len(paths)
        self.data = None
        if self.feats is None:
            self.feats = [""]
        for feat in feats:
            utils.assert_oneof(feat, self.FEATURES, "sequence feature")
        self.getdata_map = {
            feat: getattr(self, f"get_{feat}") for feat in self.FEATURES
        }
        for feat in self.getdata_map:
            utils.assert_oneof(feat, self.FEATURES)
        self._load_data()

    def _load_data(self):
        reader = TextFileReader()
        self.data = []
        for path in self.paths:
            with reader(path) as f:
                data = [line.rstrip().split() for line in f]
                if self.pad_eos is not None:
                    data = [sent + [self.pad_eos] for sent in data]
                if self.pad_bos is not None:
                    data = [[self.pad_bos] + sent for sent in data]
                self.data.append(data)
        self.data = list(zip(*self.data))

        for i in range(len(self.vocabs)):
            vocab = self.vocabs[i]
            if vocab is None:
                vocab = utils.Vocabulary()
                utils.populate_vocab(
                    words=[w for s in self.data for w in s[i]],
                    vocab=vocab,
                    cutoff=self.vocab_limit
                )
                vocab.add("<unk>")
                self.vocabs[i] = vocab
            self.unk_idxs[i] = vocab.f2i.get(self.unk)

    def _word2idx(self, i, w):
        vocab, unk_idx = self.vocabs[i], self.unk_idxs[i]
        return vocab.f2i.get(w, unk_idx)

    def get_string(self, tokens_list):
        return tuple(" ".join(tokens) for tokens in tokens_list)

    def get_tensor(self, tokens_list):
        return tuple(torch.LongTensor([self._word2idx(i, w)
                                       for w in tokens])
                     for i, tokens in enumerate(tokens_list))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        ret = dict()
        tokens_list = self.data[item]
        for feat in self.feats:
            ret[feat] = self.getdata_map[feat](tokens_list)
        return ret


class TextSequenceBatchCollator(object):
    FEATURES = {
        "string": "list",
        "tensor": "tensorvarlist"
    }
    DATA_TYPES = {
        "list",
        "tensor",
        "tensorlist",
        "tensorvar",
        "tensorvarlist"
    }

    def __init__(self, pad_idxs=0):
        self.pad_idxs = pad_idxs
        self.collate_map = {
            dt: getattr(self, f"collate_{dt}") for dt in self.DATA_TYPES
        }
        # sanity check
        for dt in self.collate_map:
            utils.assert_oneof(dt, self.DATA_TYPES)
        assert set(self.FEATURES) == set(TextSequenceDataset.FEATURES)

    def collate_list(self, batch):
        return batch

    def collate_tensor(self, batch):
        return torch.stack(batch)

    def collate_tensorvar(self, batch, pad_idx=None):
        lens = torch.LongTensor(list(map(len, batch)))
        max_len = lens.max().item()
        if pad_idx is None:
            pad_idx = self.pad_idxs[0]
        return pad_sequences(batch, max_len, pad_idx), lens

    def collate_tensorlist(self, batch):
        batch = list(zip(*batch))
        return [self.collate_tensor(b) for b in batch]

    def collate_tensorvarlist(self, batch):
        pad_idxs = self.pad_idxs
        if not isinstance(pad_idxs, collections.Sequence):
            pad_idxs = [pad_idxs] * len(batch)
        batch = list(zip(*batch))
        return [self.collate_tensorvar(b, pad_idx)
                for b, pad_idx in zip(batch, pad_idxs)]

    def __call__(self, batches):
        sample = batches[0]
        ret = dict()
        for feat in sample:
            items = [inst[feat] for inst in batches]
            dt = self.FEATURES[feat]
            collate_fn = self.collate_map[dt]
            ret[feat] = collate_fn(items)
        return ret



# dataset = TextSequenceDataset(r"D:\Downloads\SlotGated-SLU-master\SlotGated-SLU-master\data\atis\train\seq.in.txt", feats=["string", "tensor"])
# collator = TextSequenceBatchCollator(len(dataset.vocab))
# dataloader = td.DataLoader(dataset, batch_size=32, collate_fn=collator, shuffle=True)
# print(next(iter(dataloader))["tensor"][1].size())