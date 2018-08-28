import torch
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import train.embeds


class AbstractSentenceEncoder(object):
    def encode(self, sents):
        """
        encodes sentences into fixed-size vectors
        :param sents: list of strings
        :return: [num_sents x dim] tensor
        """

class UniversalSentenceEncoder(AbstractSentenceEncoder):
    def __init__(self, batch_size, minimal_gpumem=True):
        # https://www.tensorflow.org/hub/modules/google/
        # universal-sentence-encoder
        self.use_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.use = hub.Module(self.use_url)
        self.batch_size = batch_size
        self.minimal_gpumem = minimal_gpumem
        self.tf_device = "cpu:0"
        # universal sentence encoder does not support gpu specification
        # if self.device.index is None:
        #     self.tf_device = "cpu:0"
        # else:
        #     self.tf_device = f"device:GPU:{self.device.index}"

    def encode(self, sents):
        config = tf.ConfigProto()
        if self.minimal_gpumem:
            config.gpu_options.allow_growth = True
        with tf.device(f"/{self.tf_device}"), tf.Session(config=config) as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            embs = [session.run(self.use(sents[i:i + self.batch_size]))
                    for i in range(0, len(sents), self.batch_size)]
            return torch.tensor(np.concatenate(embs))


class WordEmbeddingSentenceEncoder(AbstractSentenceEncoder):
    def __init__(self, word_embs: train.embeds.Embeddings):
        self.word_embs = word_embs

    def encode_one(self, sent):
        embs = [self.word_embs[w] for w in sent.split() if w in self.word_embs]
        if not embs:
            return
        return np.stack(embs).mean(0)

    def encode(self, sents):
        vecs = [self.encode_one(sent) for sent in sents]
        vecs = [v for v in vecs if v is not None]
        return torch.tensor(np.stack(vecs))


class Analyzer(object):
    def __init__(self, encoder: AbstractSentenceEncoder, device):
        self.encoder = encoder
        self.device = device

    @staticmethod
    def intracluster_distance(vecs):
        delta = vecs - vecs.mean(0)
        dists = delta.pow(2).sum(1).sqrt()
        return dists.mean().item()

    @staticmethod
    def normalize(vecs):
        norm = vecs.pow(2).sum(1).sqrt()
        return vecs / norm.unsqueeze(-1)

    def analyze(self, samples):
        embs = self.encoder.encode(samples)
        embs = embs.to(self.device)
        embs = self.normalize(embs)
        return {
            "icd": self.intracluster_distance(embs)
        }