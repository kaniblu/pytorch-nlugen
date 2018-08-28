import torch
import torch.nn as nn

from . import common
from . import nonlinear


class BaseFusion(common.Module):
    def __init__(self, in_dims, out_dim):
        super(BaseFusion, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim

    def forward_loss(self, *xs):
        raise NotImplementedError()


class ConcatNonlinearFusion(BaseFusion):
    name = "concat-nonlinear"

    def __init__(self, *args, **kwargs):
        super(ConcatNonlinearFusion, self).__init__(*args, **kwargs)
        self.nonlinear = nonlinear.get_default()(
            in_dim=sum(self.in_dims),
            out_dim=self.out_dim,
        )

    def forward_loss(self, *xs):
        return self.invoke(self.nonlinear, torch.cat(xs, 1))


class BaseDimMatchFusion(BaseFusion):
    def __init__(self, in_dims, out_dim):
        super(BaseDimMatchFusion, self).__init__(in_dims, out_dim)
        self.in_nonlinears = common.ModuleList([
            nonlinear.get_default()(in_dim, out_dim) for in_dim in in_dims
        ])
        self.out_nonlinear = nonlinear.get_default()(out_dim)

    def fuse(self, *xs):
        raise NotImplementedError()

    def forward_loss(self, *xs):
        xs = [self.invoke(nonlinear, x)
              for nonlinear, x in zip(self.in_nonlinears, xs)]
        x = self.invoke(self.fuse, *xs)
        return self.invoke(self.out_nonlinear, x)


class MLBSumFusion(BaseDimMatchFusion):
    name = "mlb-sum"

    def fuse(self, *xs):
        return torch.stack(xs, 1).sum(1)


class MLBFusion(BaseDimMatchFusion):
    name = "mlb"

    def fuse(self, *xs):
        return torch.stack(xs, 1).prod(1)


class GatedSoftmaxFusion(BaseDimMatchFusion):
    name = "mlb-gated"

    def __init__(self, in_dims, out_dim):
        super(GatedSoftmaxFusion, self).__init__(in_dims, out_dim)
        inputs = len(in_dims)
        self.linears = common.ModuleList([
            common.Linear(out_dim * inputs, out_dim) for _ in range(inputs)
        ])
        self.softmax = nn.Softmax(1)

    def fuse(self, *xs):
        xs = torch.cat(xs, 1)
        logits = [self.invoke(linear, xs) for linear in self.linears]
        logits = torch.stack(logits, 1)
        atts = self.softmax(logits)
        return (atts * logits).sum(1)


class GatedSigmoidFusion(BaseDimMatchFusion):
    name = "mlb-gated-sigmoid"

    def __init__(self, in_dims, out_dim):
        super(GatedSigmoidFusion, self).__init__(in_dims, out_dim)
        inputs = len(in_dims)
        self.linears = common.ModuleList([
            common.Linear(out_dim * inputs, out_dim) for _ in range(inputs)
        ])
        self.sigmoid = nn.Sigmoid()

    def fuse(self, *xs):
        xs = torch.cat(xs, 1)
        logits = [self.invoke(linear, xs) for linear in self.linears]
        logits = torch.stack(logits, 1)
        atts = self.sigmoid(logits)
        return (atts * logits).sum(1)


class GeneralizedFusion(BaseDimMatchFusion):
    name = "generalized-fusion"

    def __init__(self, *args, bias=False, **kwargs):
        super(GeneralizedFusion, self).__init__(*args, **kwargs)
        inputs = len(self.in_dims)
        out_dim = self.out_dim
        self.linears = common.ModuleList([
            common.Linear(out_dim * inputs, out_dim)
            for _ in range(self.num_subfuses())
        ])
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.bias = None
        if bias:
            self.bias = common.Parameter(torch.zeros(out_dim))

    def num_subfuses(self):
        return 4

    def subfuse(self, x, y):
        return torch.stack([x, y, x - y, x * y], 1)

    def fuse(self, *xs):
        x, y = xs
        xs = torch.cat(xs, 1)
        g = torch.stack([self.invoke(linear, xs) for linear in self.linears], 1)
        h = self.subfuse(x, y)
        h = (self.sigmoid(g) * h).sum(1) + x + y
        if self.bias is not None:
            h += self.bias
        return h


class GeneralizedFusionWithAbs(GeneralizedFusion):
    name = "generalized-fusion-abs"

    def num_subfuses(self):
        return 4

    def subfuse(self, x, y):
        return torch.stack([x, y, torch.abs(x - y), x * y], 1)


class GatedBinaryFusion(BaseDimMatchFusion):
    name = "mlb-bingated"

    def __init__(self, *args, **kwargs):
        super(GatedBinaryFusion, self).__init__(*args, **kwargs)
        assert len(self.in_dims) == 2, \
            "input dimensions must contain two elements"
        self.linear = common.Linear(self.out_dim * 2, self.out_dim)
        self.sigmoid = nn.Sigmoid()

    def fuse(self, *xs):
        gate = self.sigmoid(self.invoke(self.linear, torch.cat(xs, 1)))
        return gate * xs[0] + (1 - gate) * xs[1]


MODULES = [
    GatedSoftmaxFusion,
    ConcatNonlinearFusion,
    GeneralizedFusion,
    GeneralizedFusionWithAbs,
    MLBFusion,
    MLBSumFusion,
    GatedSigmoidFusion,
    GatedBinaryFusion,
]
