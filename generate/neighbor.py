import utils
import logging
import importlib
import collections

import torch
import numpy as np


class AbstractNNSearcher(object):

    def __init__(self, sents, num_neighbors, **kwargs):
        logging.warning(f"ignored kwargs: {kwargs}")
        self.sents = sents
        self.num_neighbors = num_neighbors

    def search(self, queries):
        raise NotImplementedError()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class PyTorchPCASearcher(AbstractNNSearcher):

    def __init__(self, *args, pca_dim=10, batch_size=32, device=None,
                 bos="<bos>", eos="<eos>", unk="<unk>", **kwargs):
        super(PyTorchPCASearcher, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.device = device
        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.pca_dim = pca_dim

        if self.device is None:
            self.device = torch.device("cpu")

        self.vocab = utils.Vocabulary()
        self.vocab.add(bos)
        self.vocab.add(eos)
        self.vocab.add(unk)
        utils.populate_vocab(self.words, self.vocab)
        self.unk_idx = self.vocab.f2i[self.unk]

        self.sparse = importlib.import_module("scipy.sparse")
        self.decomp = importlib.import_module("sklearn.decomposition")
        self.sents_csr = self.sparse.vstack([self.to_csr(s) for s in self.sents])
        self.pca = self.get_pca(self.sents_csr)
        self.sents_pca = self.pca.transform(self.sents_csr)
        self.sents_tensor = torch.Tensor(self.sents_pca).to(self.device)

    def to_csr(self, sent):
        bow = collections.Counter(sent.split())
        bow = {self.vocab.f2i.get(k, self.unk_idx): v for k, v in bow.items()}
        row_idx, vals = list(zip(*bow.items()))
        return self.sparse.csr_matrix(
            (vals, ([0] * len(row_idx), row_idx)),
            shape=[1, len(self.vocab)],
            dtype=np.int8
        )

    def get_pca(self, csr):
        logging.info("performing pca...")
        pca = self.decomp.TruncatedSVD(n_components=self.pca_dim)
        pca.fit(csr)
        return pca

    @property
    def words(self):
        for s in self.sents:
            for w in s.split():
                yield w

    def search(self, queries):
        with torch.no_grad():
            neighbors = []
            num_queries = len(queries)
            for i in utils.tqdm(range(0, num_queries, self.batch_size),
                                desc="searching nearest neighbors"):
                x = queries[i:i + self.batch_size]
                x = self.sparse.vstack([self.to_csr(s) for s in x])
                x = self.pca.transform(x)
                x = torch.Tensor(x).to(self.device)
                logits = torch.matmul(self.sents_tensor, x.t()).t()
                idxs = torch.sort(logits, 1, True)[1][:, :self.num_neighbors]
                idxs = idxs.cpu().tolist()
                neighbors.extend([[self.sents[j] for j in idx] for idx in idxs])
            return neighbors


class PyTorchNNSearcher(AbstractNNSearcher):

    def __init__(self, *args, batch_size=32, device=None,
                 bos="<bos>", eos="<eos>", unk="<unk>", **kwargs):
        super(PyTorchNNSearcher, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.device = device
        self.bos = bos
        self.eos = eos
        self.unk = unk

        if self.device is None:
            self.device = torch.device("cpu")
        self.vocab = utils.Vocabulary()
        self.vocab.add(bos)
        self.vocab.add(eos)
        self.vocab.add(unk)
        utils.populate_vocab(self.words, self.vocab)
        self.tensors = torch.stack([self.tensorize_bow(s) for s in self.sents])
        self.tensors = self.tensors.to(self.device)

    @property
    def words(self):
        for s in self.sents:
            for w in s.split():
                yield w

    def tensorize_bow(self, sent):
        bow = utils.normalize(collections.Counter(sent.split()))
        bow = {k: 1 for k in bow}
        tensor = torch.zeros(len(self.vocab))
        for k, v in bow.items():
            tensor[self.vocab.f2i[k]] = v
        return tensor

    def search(self, queries):
        with torch.no_grad():
            neighbors = []
            num_queries = len(queries)
            for i in utils.tqdm(range(0, num_queries, self.batch_size),
                                desc="searching nearest neighbors"):
                x = queries[i:i + self.batch_size]
                x = torch.stack([self.tensorize_bow(s) for s in x]).to(self.device)
                logits = torch.matmul(self.tensors, x.t()).t()
                idxs = torch.sort(logits, 1, True)[1][:, :self.num_neighbors]
                idxs = idxs.cpu().tolist()
                neighbors.extend([[self.sents[j] for j in idx] for idx in idxs])
            return neighbors


class SimpleNNSearcher(AbstractNNSearcher):

    def __init__(self, *args, workers=10, **kwargs):
        super(SimpleNNSearcher, self).__init__(*args, **kwargs)
        self.workers = workers

        self.sents_bow = [self.create_bag_of_words(s) for s in self.sents]

    @staticmethod
    def create_bag_of_words(string):
        return utils.normalize(collections.Counter(string.split()))

    @staticmethod
    def bow_cosine_similarity(bow1, bow2):
        # similarity in terms of bow1
        bow2 = {k: 1 for k, _ in bow2.items()}
        common_keys = set(bow1) & set(bow2)
        return sum(bow1[k] * bow2[k] for k in common_keys)

    @staticmethod
    def mp_progress(func, iterable, processes=10, scale=10):
        gensim = utils.import_module("gensim")
        mp = utils.import_module("multiprocessing.pool")
        chunks = list(gensim.utils.chunkize(iterable, processes * scale))
        pool = mp.Pool(processes)
        ret = []

        for chunk in utils.tqdm(chunks):
            ret.extend(pool.map(func, chunk))

        return ret

    @staticmethod
    def get_knn(args):
        (sample_bow, dataset_bow, strings, k) = args
        cossim = SimpleNNSearcher.bow_cosine_similarity
        sims = [(cossim(sample_bow, dataset_bow[i]), s)
                for i, s in enumerate(strings)]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in sims[:k]]

    def search(self, queries):
        queries_bow = [self.create_bag_of_words(s) for s in queries]
        num_samples = len(queries)
        targs = list(zip(
            queries_bow,
            [self.sents_bow] * num_samples,
            [self.sents] * num_samples,
            [self.num_neighbors] * num_samples
        ))
        return self.mp_progress(self.get_knn, targs, self.workers, 10)
