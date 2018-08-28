import torch

import utils


class Encoder(object):
    def __init__(self, model, device, batch_size):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.step = None
        self.tensor_key = "tensor"

    def prepare_batch(self, batch):
        data = batch[self.tensor_key]
        data = [(x.to(self.device), lens.to(self.device)) for x, lens in
                data]
        (w, w_lens), (l, l_lens), (i, i_lens) = data
        batch_size = w.size(0)
        return batch_size, (w, l, i[:, 1], w_lens)

    def encode(self, dataloader):
        self.model.train(False)
        self.step = 0
        progress = utils.tqdm(
            total=len(dataloader.dataset),
            desc=f"encoding distribution",
        )
        means, stds = [], []
        for batch in dataloader:
            batch_size, (w, l, i, lens) = self.prepare_batch(batch)
            self.step += batch_size
            progress.update(batch_size)
            mean, std = self.model.encode(w, l, i, lens)
            means.append(mean.cpu())
            stds.append(std.cpu())
        progress.close()
        return torch.cat(means, 0), torch.cat(stds, 0)