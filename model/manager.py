import logging

import utils
from . import common


PACKAGES = {
    "rnn",
    "vae",
    "ved",
    "luvae",
    "mmvae",
    "fusion",
    "encoder",
    "decoder",
    "pooling",
    "gaussian",
    "nonlinear",
    "embedding",
    "paired_decoders",
    "discrete_decoder",
    "discrete_encoder"
}
_CLSMAP_CACHE = None


def _resolve_pkg(pkg_name):
    return utils.import_module(f"model.{pkg_name}")


def _resolve_pkg_arg(pkg):
    if pkg is None:
        pkgs = [_resolve_pkg(pkg) for pkg in PACKAGES]
    else:
        if isinstance(pkg, str):
            pkg = _resolve_pkg(pkg)
        pkgs = [pkg]
    return pkgs


def _get_classes(pkg):
    return [cls for cls in pkg.__dict__.values() if isinstance(cls, type)]


def _generate_clskvp(classes):
    for cls in classes:
        if isinstance(cls.name, tuple):
            for name in cls.name:
                yield name, cls
        else:
            yield cls.name, cls


def get_module_classes(pkg=None):
    return [cls for pkg in _resolve_pkg_arg(pkg) for cls in _get_classes(pkg)
            if issubclass(cls, common.Module) and cls.name is not None]


def get_module_names(pkg=None):
    return [name for name, _ in _generate_clskvp(get_module_classes(pkg))]


def get_module_namemap(pkg=None):
    return dict(_generate_clskvp(get_module_classes(pkg)))


def get(name, pkg=None):
    if pkg is None:
        global _CLSMAP_CACHE
        if _CLSMAP_CACHE is None:
            _CLSMAP_CACHE = get_module_namemap()
        clsmap = _CLSMAP_CACHE
    else:
        clsmap = get_module_namemap(pkg)
    if name not in clsmap:
        logging.error(f"{name} is not a recognized module. "
                      f"Did you add the parent package to 'manager.py'?")
    return clsmap.get(name)