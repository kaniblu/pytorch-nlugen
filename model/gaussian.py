import torch

from . import common
from . import nonlinear


class AbstractGaussianSampling(common.Module):

    def __init__(self, in_dim, out_dim):
        super(AbstractGaussianSampling, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward_loss(self, x):
        raise NotImplementedError()

    def mean(self, x):
        raise NotImplementedError()

    def std(self, x):
        raise NotImplementedError()


class ReparameterizedGaussianSampling(AbstractGaussianSampling):

    name = "reparam-gaussian-sampling"

    def __init__(self, *args, enforce_unit=False, loss_scale=1.0,
                 no_variance=False, **kwargs):
        super(ReparameterizedGaussianSampling, self).__init__(*args, **kwargs)
        self.enforce_unit = enforce_unit
        self.loss_scale = loss_scale
        self.no_variance = no_variance
        self.mu_nonlinear = nonlinear.get_default()(
            in_dim=self.in_dim,
            out_dim=self.out_dim
        )
        self.lv_nonlinear = nonlinear.get_default()(
            in_dim=self.in_dim,
            out_dim=self.out_dim
        )

    def sample(self, mu, logvar):
        if self.no_variance:
            return mu
        std = torch.exp(0.5 * logvar)
        rnd = torch.randn_like(std)
        return rnd * std + mu

    @staticmethod
    def kld_loss(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)

    def forward_loss(self, x):
        mu = self.invoke(self.mu_nonlinear, x)
        lv = self.invoke(self.lv_nonlinear, x)
        yield "pass", self.sample(mu, lv)
        if self.enforce_unit:
            yield "loss", self.kld_loss(mu, lv) * self.loss_scale

    def mean(self, x):
        return self.invoke(self.mu_nonlinear, x)

    def std(self, x):
        lv = self.invoke(self.lv_nonlinear, x)
        return torch.exp(0.5 * lv)


class BugFixedRGaussianSampling(AbstractGaussianSampling):

    name = "bug-fixed-reparam-gaussian-sampling"

    def __init__(self, *args, enforce_unit=False, loss_scale=1.0,
                 no_variance=False, **kwargs):
        super(BugFixedRGaussianSampling, self).__init__(*args, **kwargs)
        self.enforce_unit = enforce_unit
        self.loss_scale = loss_scale
        self.no_variance = no_variance
        self.mu_nonlinear = nonlinear.get_default()(
            in_dim=self.in_dim,
            out_dim=self.out_dim
        )
        self.mu_linear = common.Linear(self.out_dim, self.out_dim)
        self.lv_nonlinear = nonlinear.get_default()(
            in_dim=self.in_dim,
            out_dim=self.out_dim
        )
        self.lv_linear = common.Linear(self.out_dim, self.out_dim)

    def sample(self, mu, logvar):
        if self.no_variance:
            return mu
        std = torch.exp(0.5 * logvar)
        rnd = torch.randn_like(std)
        return rnd * std + mu

    @staticmethod
    def kld_loss(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)

    def forward_loss(self, x):
        mu = self.invoke(self.mu_linear, self.invoke(self.mu_nonlinear, x))
        lv = self.invoke(self.lv_linear, self.invoke(self.lv_nonlinear, x))
        yield "pass", self.sample(mu, lv)
        if self.enforce_unit:
            yield "loss", self.kld_loss(mu, lv) * self.loss_scale

    def mean(self, x):
        return self.invoke(self.mu_linear, self.invoke(self.mu_nonlinear, x))

    def std(self, x):
        lv = self.invoke(self.lv_linear, self.invoke(self.lv_nonlinear, x))
        return torch.exp(0.5 * lv)